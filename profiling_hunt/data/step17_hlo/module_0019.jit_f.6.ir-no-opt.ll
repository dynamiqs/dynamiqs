; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @loop_add_fusion_5(ptr noalias align 256 dereferenceable(8) %0, ptr noalias align 256 dereferenceable(8) %1) #0 {
  %3 = getelementptr inbounds [1 x i64], ptr %0, i32 0, i32 0
  %4 = load i64, ptr %3, align 4
  %5 = add i64 %4, 1
  store i64 %5, ptr %3, align 4
  ret void
}

attributes #0 = { "nvvm.reqntid"="1,1,1" }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
