; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @wrapped_or(ptr noalias align 16 dereferenceable(1) %0, ptr noalias align 16 dereferenceable(1) %1, ptr noalias align 256 dereferenceable(1) %2) #0 {
  %4 = getelementptr inbounds [1 x i8], ptr %0, i32 0, i32 0
  %5 = load i8, ptr %4, align 1, !invariant.load !1
  %6 = getelementptr inbounds [1 x i8], ptr %1, i32 0, i32 0
  %7 = load i8, ptr %6, align 1, !invariant.load !1
  %8 = or i8 %5, %7
  %9 = getelementptr inbounds [1 x i8], ptr %2, i32 0, i32 0
  store i8 %8, ptr %9, align 1
  ret void
}

attributes #0 = { "nvvm.reqntid"="1,1,1" }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{}
