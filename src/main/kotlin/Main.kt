package org.example

import ai.djl.Device
import ai.djl.engine.Engine
import ai.djl.ndarray.NDManager

/**おｒｇ。えぁｍｐぇ
 * Minimal program to check GPU recognition status and operation in DJL
 */
fun main() {
    // 1. Check the number of GPUs recognized by the system
    val gpuCount = Engine.getInstance().gpuCount
    println("Number of GPUs recognized: $gpuCount")

    if (gpuCount > 0) {
        // Get the default GPU device
        val device = Device.gpu()
        println("Device to be used: $device")

        // 2. Create an NDManager associated with the GPU device
        // Data created through this manager will be placed in GPU memory
        NDManager.newBaseManager(device).use { manager ->
            // 3. Create an NDArray on the GPU (vector of 1.0, 2.0, 3.0)
            val array = manager.create(floatArrayOf(1.0f, 2.0f, 3.0f))

            // 4. Display and verify the device where the array is actually placed
            println("Device where the array is placed: ${array.device}")

            if (array.device.isGpu) {
                println("Status: GPU is operating normally.")
            } else {
                println("Status: GPU was specified but fell back to CPU. Please check your configuration.")
            }
        }
    } else {
        println("Status: No GPU recognized by the system. Only CPU is available.")
        println("Hint: Please check if CUDA drivers and DJL GPU native dependencies (e.g., pytorch-native-cuXXX) are correct.")
    }
}
