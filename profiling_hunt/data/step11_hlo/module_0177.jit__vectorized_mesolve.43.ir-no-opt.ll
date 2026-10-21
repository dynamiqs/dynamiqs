; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @loop_add_fusion_2(ptr noalias align 256 dereferenceable(8) %0, ptr noalias align 256 dereferenceable(1) %1, ptr noalias align 256 dereferenceable(8) %2) #0 {
  %4 = getelementptr inbounds [1 x i8], ptr %1, i32 0, i32 0
  %5 = load i8, ptr %4, align 1, !invariant.load !1
  %6 = getelementptr inbounds [1 x i64], ptr %0, i32 0, i32 0
  %7 = load i64, ptr %6, align 4
  %8 = trunc i8 %5 to i1
  %9 = xor i1 %8, true
  %10 = zext i1 %9 to i64
  %11 = add i64 %7, %10
  store i64 %11, ptr %6, align 4
  ret void
}

attributes #0 = { "nvvm.reqntid"="1,1,1" }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{}
