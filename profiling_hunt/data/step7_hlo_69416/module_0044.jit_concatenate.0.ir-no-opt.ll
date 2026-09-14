; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @wrapped_concatenate(ptr noalias align 16 dereferenceable(4) %0, ptr noalias align 16 dereferenceable(4) %1, ptr noalias align 256 dereferenceable(8) %2) #0 {
  %4 = getelementptr inbounds [1 x i32], ptr %0, i32 0, i32 0
  %5 = load i32, ptr %4, align 4, !invariant.load !1
  %6 = getelementptr inbounds [2 x i32], ptr %2, i32 0, i32 0
  store i32 %5, ptr %6, align 4
  %7 = getelementptr inbounds [1 x i32], ptr %1, i32 0, i32 0
  %8 = load i32, ptr %7, align 4, !invariant.load !1
  %9 = getelementptr inbounds [2 x i32], ptr %2, i32 0, i32 1
  store i32 %8, ptr %9, align 4
  ret void
}

attributes #0 = { "nvvm.reqntid"="1,1,1" }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{}
