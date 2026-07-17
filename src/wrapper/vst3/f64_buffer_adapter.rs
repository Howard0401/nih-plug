//! Allocation-free VST3 `kSample64` boundary conversion for the f32 plugin API.

use std::num::NonZeroU32;
use std::ptr::NonNull;

use phaselith_vst3_sys::vst::{AudioBusBuffers, ProcessData};

use crate::prelude::AudioIOLayout;
use crate::wrapper::util::buffer_management::ChannelPointers;

struct F32BusScratch {
    channels: Vec<Vec<f32>>,
    pointers: Vec<*mut f32>,
}

impl F32BusScratch {
    fn new(num_channels: usize, max_samples: usize) -> Self {
        let mut channels = vec![vec![0.0; max_samples]; num_channels];
        let pointers = channels
            .iter_mut()
            .map(|channel| channel.as_mut_ptr())
            .collect();

        Self { channels, pointers }
    }

    fn clear(&mut self, num_samples: usize) {
        for channel in &mut self.channels {
            channel[..num_samples].fill(0.0);
        }
    }

    unsafe fn load_f64(&mut self, bus: &AudioBusBuffers, num_samples: usize) {
        self.clear(num_samples);
        if bus.buffers.is_null() || bus.num_channels <= 0 {
            return;
        }

        let host_pointers = bus.buffers.cast::<*mut f64>();
        let copied_channels = self.channels.len().min(bus.num_channels as usize);
        for channel_idx in 0..copied_channels {
            let host_channel = *host_pointers.add(channel_idx);
            if host_channel.is_null() {
                continue;
            }

            for (dst, src) in self.channels[channel_idx][..num_samples]
                .iter_mut()
                .zip(std::slice::from_raw_parts(host_channel, num_samples))
            {
                *dst = *src as f32;
            }
        }
    }

    unsafe fn store_f64(
        &self,
        bus: &AudioBusBuffers,
        num_samples: usize,
        residual: Option<&[Vec<f64>]>,
    ) {
        if bus.buffers.is_null() || bus.num_channels <= 0 {
            return;
        }

        let host_pointers = bus.buffers.cast::<*mut f64>();
        let copied_channels = self.channels.len().min(bus.num_channels as usize);
        for channel_idx in 0..copied_channels {
            let host_channel = *host_pointers.add(channel_idx);
            if host_channel.is_null() {
                continue;
            }

            let host_channel = std::slice::from_raw_parts_mut(host_channel, num_samples);
            for sample_idx in 0..num_samples {
                let carrier_residual = residual
                    .and_then(|channels| channels.get(channel_idx))
                    .map(|channel| channel[sample_idx])
                    .unwrap_or(0.0);
                host_channel[sample_idx] =
                    f64::from(self.channels[channel_idx][sample_idx]) + carrier_residual;
            }
        }
    }

    fn channel_pointers(&mut self) -> ChannelPointers {
        ChannelPointers {
            ptrs: NonNull::new(self.pointers.as_mut_ptr()).expect("non-empty audio bus"),
            num_channels: self.channels.len(),
        }
    }
}

struct ResidualDelay {
    lines: Vec<Vec<f64>>,
    write_positions: Vec<usize>,
    delayed_block: Vec<Vec<f64>>,
    capacity: usize,
}

impl ResidualDelay {
    fn new(num_channels: usize, max_samples: usize, capacity: usize) -> Self {
        Self {
            lines: vec![vec![0.0; capacity]; num_channels],
            write_positions: vec![0; num_channels],
            delayed_block: vec![vec![0.0; max_samples]; num_channels],
            capacity,
        }
    }

    fn prepare(
        &mut self,
        quantized_input: Option<&F32BusScratch>,
        host_input: Option<&AudioBusBuffers>,
        num_samples: usize,
        latency_samples: usize,
    ) {
        for channel in &mut self.delayed_block {
            channel[..num_samples].fill(0.0);
        }

        let Some(quantized_input) = quantized_input else {
            return;
        };
        let Some(host_input) = host_input else {
            self.push_silence(num_samples, latency_samples);
            return;
        };
        if host_input.buffers.is_null() || host_input.num_channels <= 0 {
            self.push_silence(num_samples, latency_samples);
            return;
        }

        let host_pointers = host_input.buffers.cast::<*mut f64>();
        let host_channels = host_input.num_channels as usize;
        let latency_is_covered = latency_samples < self.capacity;

        for channel_idx in 0..self.lines.len() {
            let host_channel = if channel_idx < host_channels {
                unsafe { *host_pointers.add(channel_idx) }
            } else {
                std::ptr::null_mut()
            };
            let mut write_position = self.write_positions[channel_idx];

            for sample_idx in 0..num_samples {
                let residual =
                    if host_channel.is_null() || channel_idx >= quantized_input.channels.len() {
                        0.0
                    } else {
                        let host_sample = unsafe { *host_channel.add(sample_idx) };
                        let quantized_sample =
                            f64::from(quantized_input.channels[channel_idx][sample_idx]);
                        if host_sample.is_finite() && quantized_sample.is_finite() {
                            host_sample - quantized_sample
                        } else {
                            0.0
                        }
                    };

                self.lines[channel_idx][write_position] = residual;
                if latency_is_covered {
                    let read_position =
                        (write_position + self.capacity - latency_samples) % self.capacity;
                    self.delayed_block[channel_idx][sample_idx] =
                        self.lines[channel_idx][read_position];
                }
                write_position += 1;
                if write_position == self.capacity {
                    write_position = 0;
                }
            }

            self.write_positions[channel_idx] = write_position;
        }
    }

    fn push_silence(&mut self, num_samples: usize, latency_samples: usize) {
        let latency_is_covered = latency_samples < self.capacity;
        for channel_idx in 0..self.lines.len() {
            let mut write_position = self.write_positions[channel_idx];
            for sample_idx in 0..num_samples {
                self.lines[channel_idx][write_position] = 0.0;
                if latency_is_covered {
                    let read_position =
                        (write_position + self.capacity - latency_samples) % self.capacity;
                    self.delayed_block[channel_idx][sample_idx] =
                        self.lines[channel_idx][read_position];
                }
                write_position += 1;
                if write_position == self.capacity {
                    write_position = 0;
                }
            }
            self.write_positions[channel_idx] = write_position;
        }
    }
}

/// Preallocated f64 host boundary storage. All vector sizes are fixed before activation.
pub(crate) struct F64BufferAdapter {
    max_samples: usize,
    main_input: Option<F32BusScratch>,
    main_output: Option<F32BusScratch>,
    aux_inputs: Vec<F32BusScratch>,
    aux_outputs: Vec<F32BusScratch>,
    main_input_present: bool,
    main_output_present: bool,
    aux_inputs_present: Vec<bool>,
    aux_outputs_present: Vec<bool>,
    residual_delay: ResidualDelay,
}

// SAFETY: Raw pointers in each `F32BusScratch` point only into its own fixed-capacity vectors.
// The adapter is exclusively borrowed on the audio thread while those pointers are exposed.
unsafe impl Send for F64BufferAdapter {}
unsafe impl Sync for F64BufferAdapter {}

impl F64BufferAdapter {
    pub fn for_audio_io_layout(
        max_samples: usize,
        audio_io_layout: AudioIOLayout,
        sample_rate: f32,
        max_latency_seconds: f64,
    ) -> Self {
        let channel_count =
            |channels: Option<NonZeroU32>| channels.map(NonZeroU32::get).unwrap_or(0) as usize;
        let main_input_channels = channel_count(audio_io_layout.main_input_channels);
        let main_output_channels = channel_count(audio_io_layout.main_output_channels);
        let residual_channels = main_input_channels.min(main_output_channels);
        let bounded_seconds = if max_latency_seconds.is_finite() && max_latency_seconds >= 0.0 {
            max_latency_seconds
        } else {
            0.0
        };
        let residual_capacity =
            ((f64::from(sample_rate) * bounded_seconds).ceil() as usize).saturating_add(1);

        Self {
            max_samples,
            main_input: (main_input_channels > 0)
                .then(|| F32BusScratch::new(main_input_channels, max_samples)),
            main_output: (main_output_channels > 0)
                .then(|| F32BusScratch::new(main_output_channels, max_samples)),
            aux_inputs: audio_io_layout
                .aux_input_ports
                .iter()
                .map(|channels| F32BusScratch::new(channels.get() as usize, max_samples))
                .collect(),
            aux_outputs: audio_io_layout
                .aux_output_ports
                .iter()
                .map(|channels| F32BusScratch::new(channels.get() as usize, max_samples))
                .collect(),
            main_input_present: false,
            main_output_present: false,
            aux_inputs_present: vec![false; audio_io_layout.aux_input_ports.len()],
            aux_outputs_present: vec![false; audio_io_layout.aux_output_ports.len()],
            residual_delay: ResidualDelay::new(
                residual_channels,
                max_samples,
                residual_capacity.max(1),
            ),
        }
    }

    pub unsafe fn prepare(&mut self, data: &ProcessData, num_samples: usize, latency_samples: u32) {
        nih_debug_assert!(num_samples <= self.max_samples);
        nih_debug_assert!(
            (latency_samples as usize) < self.residual_delay.capacity,
            "The plugin reported {} samples of latency, exceeding its declared VST3 sample64 residual bound of {} samples",
            latency_samples,
            self.residual_delay.capacity.saturating_sub(1)
        );
        self.main_input_present = false;
        self.main_output_present = false;
        self.aux_inputs_present.fill(false);
        self.aux_outputs_present.fill(false);

        let has_main_input = self.main_input.is_some();
        let has_main_output = self.main_output.is_some();
        let aux_input_start = usize::from(has_main_input);
        let aux_output_start = usize::from(has_main_output);

        let main_host_input = host_bus(data.inputs, data.num_inputs, 0).filter(|_| has_main_input);
        if let (Some(scratch), Some(host_bus)) = (self.main_input.as_mut(), main_host_input) {
            scratch.load_f64(host_bus, num_samples);
            self.main_input_present = true;
        } else if let Some(scratch) = self.main_input.as_mut() {
            scratch.clear(num_samples);
        }

        let main_host_output =
            host_bus(data.outputs, data.num_outputs, 0).filter(|_| has_main_output);
        if let (Some(scratch), Some(_)) = (self.main_output.as_mut(), main_host_output) {
            scratch.clear(num_samples);
            self.main_output_present = true;
        } else if let Some(scratch) = self.main_output.as_mut() {
            scratch.clear(num_samples);
        }

        for (index, scratch) in self.aux_inputs.iter_mut().enumerate() {
            if let Some(host_bus) = host_bus(data.inputs, data.num_inputs, index + aux_input_start)
            {
                scratch.load_f64(host_bus, num_samples);
                self.aux_inputs_present[index] = true;
            } else {
                scratch.clear(num_samples);
            }
        }
        for (index, scratch) in self.aux_outputs.iter_mut().enumerate() {
            if host_bus(data.outputs, data.num_outputs, index + aux_output_start).is_some() {
                self.aux_outputs_present[index] = true;
            }
            scratch.clear(num_samples);
        }

        self.residual_delay.prepare(
            self.main_input.as_ref(),
            main_host_input,
            num_samples,
            latency_samples as usize,
        );
    }

    pub unsafe fn write_back(&self, data: &ProcessData, num_samples: usize) {
        let has_main_output = self.main_output.is_some();
        let aux_output_start = usize::from(has_main_output);
        if let (Some(scratch), Some(host_bus)) = (
            self.main_output.as_ref(),
            host_bus(data.outputs, data.num_outputs, 0).filter(|_| has_main_output),
        ) {
            scratch.store_f64(
                host_bus,
                num_samples,
                Some(&self.residual_delay.delayed_block),
            );
        }

        for (index, scratch) in self.aux_outputs.iter().enumerate() {
            if let Some(host_bus) =
                host_bus(data.outputs, data.num_outputs, index + aux_output_start)
            {
                scratch.store_f64(host_bus, num_samples, None);
            }
        }
    }

    pub fn main_input_channel_pointers(&mut self) -> Option<ChannelPointers> {
        self.main_input_present
            .then(|| self.main_input.as_mut().unwrap().channel_pointers())
    }

    pub fn main_output_channel_pointers(&mut self) -> Option<ChannelPointers> {
        self.main_output_present
            .then(|| self.main_output.as_mut().unwrap().channel_pointers())
    }

    pub fn aux_input_channel_pointers(&mut self, index: usize) -> Option<ChannelPointers> {
        self.aux_inputs_present[index].then(|| self.aux_inputs[index].channel_pointers())
    }

    pub fn aux_output_channel_pointers(&mut self, index: usize) -> Option<ChannelPointers> {
        self.aux_outputs_present[index].then(|| self.aux_outputs[index].channel_pointers())
    }
}

unsafe fn host_bus<'a>(
    busses: *mut AudioBusBuffers,
    num_busses: i32,
    index: usize,
) -> Option<&'a AudioBusBuffers> {
    if busses.is_null() || num_busses <= 0 || index >= num_busses as usize {
        None
    } else {
        let bus = &*busses.add(index);
        (!bus.buffers.is_null() && bus.num_channels > 0).then_some(bus)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::new_nonzero_u32;
    use phaselith_vst3_sys::vst::SymbolicSampleSizes;

    unsafe fn process_data(
        input: &mut AudioBusBuffers,
        output: &mut AudioBusBuffers,
        num_samples: usize,
    ) -> ProcessData {
        let mut data: ProcessData = std::mem::zeroed();
        data.symbolic_sample_size = SymbolicSampleSizes::kSample64 as i32;
        data.num_samples = num_samples as i32;
        data.num_inputs = 1;
        data.num_outputs = 1;
        data.inputs = input;
        data.outputs = output;
        data
    }

    #[test]
    fn zero_latency_transparent_path_restores_f64_carrier_exactly() {
        let layout = AudioIOLayout {
            main_input_channels: Some(new_nonzero_u32(2)),
            main_output_channels: Some(new_nonzero_u32(2)),
            ..AudioIOLayout::const_default()
        };
        let mut adapter = F64BufferAdapter::for_audio_io_layout(4, layout, 48_000.0, 1.0);
        let mut input_l = [0.123_456_789_012_345, -0.75, 1.0e-20, 0.0];
        let mut input_r = [-0.333_333_333_333, 0.5, -1.0e-18, 1.0];
        let expected_l = input_l;
        let expected_r = input_r;
        let mut output_l = [0.0; 4];
        let mut output_r = [0.0; 4];
        let mut input_ptrs = [input_l.as_mut_ptr(), input_r.as_mut_ptr()];
        let mut output_ptrs = [output_l.as_mut_ptr(), output_r.as_mut_ptr()];
        let mut input_bus = AudioBusBuffers {
            num_channels: 2,
            silence_flags: 0,
            buffers: input_ptrs.as_mut_ptr().cast(),
        };
        let mut output_bus = AudioBusBuffers {
            num_channels: 2,
            silence_flags: 0,
            buffers: output_ptrs.as_mut_ptr().cast(),
        };
        let data = unsafe { process_data(&mut input_bus, &mut output_bus, 4) };

        unsafe { adapter.prepare(&data, 4, 0) };
        let input = adapter.main_input.as_ref().unwrap();
        let output = adapter.main_output.as_mut().unwrap();
        for channel_idx in 0..2 {
            output.channels[channel_idx].copy_from_slice(&input.channels[channel_idx]);
        }
        unsafe { adapter.write_back(&data, 4) };

        assert_eq!(output_l, expected_l);
        assert_eq!(output_r, expected_r);
    }

    #[test]
    fn reported_latency_delays_the_f64_residual_across_blocks() {
        let layout = AudioIOLayout {
            main_input_channels: Some(new_nonzero_u32(1)),
            main_output_channels: Some(new_nonzero_u32(1)),
            ..AudioIOLayout::const_default()
        };
        let mut adapter = F64BufferAdapter::for_audio_io_layout(2, layout, 48_000.0, 1.0);
        let source = [0.123_456_789_012_345, -0.333_333_333_333, 0.25, -0.5];
        let mut delayed_quantized = [0.0_f32; 4];
        for index in 2..4 {
            delayed_quantized[index] = source[index - 2] as f32;
        }
        let mut rendered = [0.0_f64; 4];

        for block in 0..2 {
            let start = block * 2;
            let mut input = [source[start], source[start + 1]];
            let mut output = [0.0_f64; 2];
            let mut input_ptrs = [input.as_mut_ptr()];
            let mut output_ptrs = [output.as_mut_ptr()];
            let mut input_bus = AudioBusBuffers {
                num_channels: 1,
                silence_flags: 0,
                buffers: input_ptrs.as_mut_ptr().cast(),
            };
            let mut output_bus = AudioBusBuffers {
                num_channels: 1,
                silence_flags: 0,
                buffers: output_ptrs.as_mut_ptr().cast(),
            };
            let data = unsafe { process_data(&mut input_bus, &mut output_bus, 2) };

            unsafe { adapter.prepare(&data, 2, 2) };
            adapter.main_output.as_mut().unwrap().channels[0][..2]
                .copy_from_slice(&delayed_quantized[start..start + 2]);
            unsafe { adapter.write_back(&data, 2) };
            rendered[start..start + 2].copy_from_slice(&output);
        }

        assert_eq!(rendered[0], 0.0);
        assert_eq!(rendered[1], 0.0);
        assert_eq!(rendered[2], source[0]);
        assert_eq!(rendered[3], source[1]);
    }
}
