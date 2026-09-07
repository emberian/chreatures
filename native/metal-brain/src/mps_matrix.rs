use metal::objc::{class, msg_send, runtime::Object, sel, sel_impl};
use metal::{mps, BufferRef, CommandBufferRef, DeviceRef, NSUInteger};
use std::ffi::CStr;
use std::mem::size_of;

const MPS_DATA_TYPE_FLOAT32: NSUInteger = 0x10000000 | 32;

#[derive(Debug)]
struct OwnedObject(*mut Object);

impl OwnedObject {
    fn get(&self) -> *mut Object {
        self.0
    }
}

pub fn command_buffer_error(command_buffer: &CommandBufferRef) -> Option<String> {
    unsafe {
        let error: *mut Object = msg_send![command_buffer, error];
        if error.is_null() {
            return None;
        }
        let description: *mut Object = msg_send![error, localizedDescription];
        if description.is_null() {
            return Some("unknown Metal command-buffer error".into());
        }
        let bytes: *const std::ffi::c_char = msg_send![description, UTF8String];
        if bytes.is_null() {
            Some("unprintable Metal command-buffer error".into())
        } else {
            Some(CStr::from_ptr(bytes).to_string_lossy().into_owned())
        }
    }
}

pub fn command_buffer_gpu_milliseconds(command_buffer: &CommandBufferRef) -> f64 {
    unsafe {
        let start: f64 = msg_send![command_buffer, GPUStartTime];
        let end: f64 = msg_send![command_buffer, GPUEndTime];
        (end - start).max(0.0) * 1000.0
    }
}

pub fn recommended_row_bytes(columns: usize) -> usize {
    unsafe {
        msg_send![class!(MPSMatrixDescriptor),
            rowBytesForColumns: columns as NSUInteger
            dataType: MPS_DATA_TYPE_FLOAT32]
    }
}

impl Drop for OwnedObject {
    fn drop(&mut self) {
        unsafe {
            let _: () = msg_send![self.0, release];
        }
    }
}

pub struct MatrixMultiplication {
    kernel: OwnedObject,
    left: OwnedObject,
    right: OwnedObject,
    result: OwnedObject,
}

impl MatrixMultiplication {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &DeviceRef,
        left_buffer: &BufferRef,
        left_rows: usize,
        left_columns: usize,
        left_row_bytes: usize,
        left_offset: usize,
        right_buffer: &BufferRef,
        result_columns: usize,
        right_row_bytes: usize,
        right_offset: usize,
        result_buffer: &BufferRef,
        result_row_bytes: usize,
    ) -> Result<Self, String> {
        if !mps::mps_supports_device(device) {
            return Err("MetalPerformanceShaders does not support this device".into());
        }
        let required_bytes = |rows: usize,
                              columns: usize,
                              row_bytes: usize,
                              offset: usize,
                              buffer: &BufferRef|
         -> Result<(), String> {
            let row_width = columns
                .checked_mul(size_of::<f32>())
                .ok_or_else(|| "MPS matrix row width overflow".to_string())?;
            if row_bytes < row_width || row_bytes % size_of::<f32>() != 0 {
                return Err("MPS matrix row stride is invalid".into());
            }
            let bytes = rows
                .checked_sub(1)
                .and_then(|x| x.checked_mul(row_bytes))
                .and_then(|x| x.checked_add(row_width))
                .and_then(|x| x.checked_add(offset))
                .ok_or_else(|| "MPS matrix extent overflow".to_string())?;
            if offset % size_of::<f32>() != 0 || bytes as u64 > buffer.length() {
                return Err("MPS matrix extent exceeds its Metal buffer".into());
            }
            Ok(())
        };
        required_bytes(
            left_rows,
            left_columns,
            left_row_bytes,
            left_offset,
            left_buffer,
        )?;
        required_bytes(
            left_columns,
            result_columns,
            right_row_bytes,
            right_offset,
            right_buffer,
        )?;
        required_bytes(
            left_rows,
            result_columns,
            result_row_bytes,
            0,
            result_buffer,
        )?;
        unsafe {
            let descriptor = |rows: usize, columns: usize, row_bytes: usize| -> *mut Object {
                msg_send![class!(MPSMatrixDescriptor),
                    matrixDescriptorWithRows: rows as NSUInteger
                    columns: columns as NSUInteger
                    rowBytes: row_bytes as NSUInteger
                    dataType: MPS_DATA_TYPE_FLOAT32]
            };
            let matrix = |buffer: &BufferRef,
                          offset: usize,
                          rows: usize,
                          columns: usize,
                          row_bytes: usize|
             -> Result<OwnedObject, String> {
                let description = descriptor(rows, columns, row_bytes);
                if description.is_null() {
                    return Err("MPSMatrixDescriptor construction failed".into());
                }
                let allocated: *mut Object = msg_send![class!(MPSMatrix), alloc];
                let initialized: *mut Object = msg_send![allocated,
                    initWithBuffer: buffer
                    offset: offset as NSUInteger
                    descriptor: description];
                if initialized.is_null() {
                    Err("MPSMatrix construction failed".into())
                } else {
                    Ok(OwnedObject(initialized))
                }
            };

            let left = matrix(
                left_buffer,
                left_offset,
                left_rows,
                left_columns,
                left_row_bytes,
            )?;
            let right = matrix(
                right_buffer,
                right_offset,
                left_columns,
                result_columns,
                right_row_bytes,
            )?;
            let result = matrix(
                result_buffer,
                0,
                left_rows,
                result_columns,
                result_row_bytes,
            )?;
            let allocated: *mut Object = msg_send![class!(MPSMatrixMultiplication), alloc];
            let kernel: *mut Object = msg_send![allocated,
                initWithDevice: device
                transposeLeft: false
                transposeRight: false
                resultRows: left_rows as NSUInteger
                resultColumns: result_columns as NSUInteger
                interiorColumns: left_columns as NSUInteger
                alpha: 1.0f64
                beta: 0.0f64];
            if kernel.is_null() {
                return Err("MPSMatrixMultiplication construction failed".into());
            }
            Ok(Self {
                kernel: OwnedObject(kernel),
                left,
                right,
                result,
            })
        }
    }

    pub fn encode(&self, command_buffer: &CommandBufferRef) {
        unsafe {
            let _: () = msg_send![self.kernel.get(),
                encodeToCommandBuffer: command_buffer
                leftMatrix: self.left.get()
                rightMatrix: self.right.get()
                resultMatrix: self.result.get()];
        }
    }
}
