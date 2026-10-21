; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @input_concatenate_fusion(ptr noalias align 16 dereferenceable(8) %0, ptr noalias align 256 dereferenceable(8) %1) #0 {
  %3 = getelementptr inbounds [1 x i64], ptr %0, i32 0, i32 0
  %4 = load i64, ptr %3, align 4, !invariant.load !1
  %5 = lshr i64 %4, 32
  %6 = trunc i64 %5 to i32
  %7 = getelementptr inbounds [2 x i32], ptr %1, i32 0, i32 0
  store i32 %6, ptr %7, align 4
  %8 = and i64 %4, 4294967295
  %9 = trunc i64 %8 to i32
  %10 = getelementptr inbounds [2 x i32], ptr %1, i32 0, i32 1
  store i32 %9, ptr %10, align 4
  ret void
}

attributes #0 = { "nvvm.reqntid"="1,1,1" }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{}
