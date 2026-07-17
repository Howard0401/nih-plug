use super::Plugin;
use crate::prelude::Vst3SubCategory;

/// Provides auxiliary metadata needed for a VST3 plugin.
pub trait Vst3Plugin: Plugin {
    /// The unique class ID that identifies this particular plugin. You can use the
    /// `*b"fooofooofooofooo"` syntax for this.
    ///
    /// This will be shuffled into a different byte order on Windows for project-compatibility.
    const VST3_CLASS_ID: [u8; 16];
    /// One or more subcategories. The host may use these to categorize the plugin. Internally this
    /// slice will be converted to a string where each character is separated by a pipe character
    /// (`|`). This string has a limit of 127 characters, and anything longer than that will be
    /// truncated.
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory];

    /// Whether this plugin accepts VST3 `kSample64` host buffers.
    ///
    /// NIH-plug's public processing API remains `f32`. Opting in enables the wrapper's
    /// allocation-free f64 boundary bridge: host samples are quantized once for the `f32` DSP
    /// core and the discarded input residual is delayed by the plugin-reported latency before it
    /// is added back to the f64 output. This makes a latency-correct transparent path preserve the
    /// host's f64 carrier while keeping the core's actual arithmetic precision explicit.
    const VST3_SUPPORTS_SAMPLE64: bool = false;

    /// Maximum latency covered by the f64 boundary bridge's residual delay, in seconds.
    ///
    /// Storage is allocated during activation, never in `process()`. A plugin opting into
    /// `VST3_SUPPORTS_SAMPLE64` must set this high enough for every latency it can report. The
    /// default one-second bound is deliberately conservative for effects while avoiding an
    /// unbounded realtime allocation contract.
    const VST3_SAMPLE64_MAX_LATENCY_SECONDS: f64 = 1.0;

    /// [`VST3_CLASS_ID`][Self::VST3_CLASS_ID`] in the correct order for the current platform so
    /// projects and presets can be shared between platforms. This should not be overridden.
    const PLATFORM_VST3_CLASS_ID: [u8; 16] = swap_vst3_uid_byte_order(Self::VST3_CLASS_ID);
}

#[cfg(not(target_os = "windows"))]
const fn swap_vst3_uid_byte_order(uid: [u8; 16]) -> [u8; 16] {
    uid
}

#[cfg(target_os = "windows")]
const fn swap_vst3_uid_byte_order(mut uid: [u8; 16]) -> [u8; 16] {
    // No mutable references in const functions, so we can't use `uid.swap()`
    let original_uid = uid;

    uid[0] = original_uid[3];
    uid[1] = original_uid[2];
    uid[2] = original_uid[1];
    uid[3] = original_uid[0];

    uid[4] = original_uid[5];
    uid[5] = original_uid[4];
    uid[6] = original_uid[7];
    uid[7] = original_uid[6];

    uid
}
