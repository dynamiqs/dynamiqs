; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @loop_and_fusion(ptr noalias align 256 dereferenceable(8) %0, ptr noalias align 256 dereferenceable(8) %1, ptr noalias align 256 dereferenceable(1) %2) #0 {
  %4 = getelementptr inbounds [1 x double], ptr %1, i32 0, i32 0
  %5 = load double, ptr %4, align 8, !invariant.load !1
  %6 = getelementptr inbounds [1 x i64], ptr %0, i32 0, i32 0
  %7 = load i64, ptr %6, align 4, !invariant.load !1
  %8 = fcmp olt double %5, 1.000000e-01
  %9 = icmp slt i64 %7, 100000
  %10 = and i1 %8, %9
  %11 = zext i1 %10 to i8
  %12 = getelementptr inbounds [1 x i8], ptr %2, i32 0, i32 0
  store i8 %11, ptr %12, align 1
  ret void
}

attributes #0 = { "nvvm.reqntid"="1,1,1" }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{}
