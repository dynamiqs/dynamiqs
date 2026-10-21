; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @wrapped_select(ptr noalias align 16 dereferenceable(1) %0, ptr noalias align 16 dereferenceable(8) %1, ptr noalias align 16 dereferenceable(8) %2, ptr noalias align 256 dereferenceable(8) %3) #0 {
  %5 = getelementptr inbounds [1 x i8], ptr %0, i32 0, i32 0
  %6 = load i8, ptr %5, align 1, !invariant.load !1
  %7 = getelementptr inbounds [1 x i64], ptr %1, i32 0, i32 0
  %8 = load i64, ptr %7, align 4, !invariant.load !1
  %9 = getelementptr inbounds [1 x i64], ptr %2, i32 0, i32 0
  %10 = load i64, ptr %9, align 4, !invariant.load !1
  %11 = trunc i8 %6 to i1
  %12 = select i1 %11, i64 %8, i64 %10
  %13 = getelementptr inbounds [1 x i64], ptr %3, i32 0, i32 0
  store i64 %12, ptr %13, align 4
  ret void
}

attributes #0 = { "nvvm.reqntid"="1,1,1" }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{}
