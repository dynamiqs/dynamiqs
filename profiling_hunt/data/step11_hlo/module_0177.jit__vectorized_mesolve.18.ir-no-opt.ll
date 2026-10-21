; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @loop_and_fusion(ptr noalias align 256 dereferenceable(8) %0, ptr noalias align 256 dereferenceable(8) %1, ptr noalias align 256 dereferenceable(8) %2, ptr noalias align 256 dereferenceable(1) %3) #0 {
  %5 = getelementptr inbounds [1 x double], ptr %1, i32 0, i32 0
  %6 = load double, ptr %5, align 8, !invariant.load !1
  %7 = getelementptr inbounds [1 x double], ptr %2, i32 0, i32 0
  %8 = load double, ptr %7, align 8, !invariant.load !1
  %9 = getelementptr inbounds [1 x i64], ptr %0, i32 0, i32 0
  %10 = load i64, ptr %9, align 4, !invariant.load !1
  %11 = fcmp olt double %6, %8
  %12 = icmp slt i64 %10, 100000
  %13 = and i1 %11, %12
  %14 = zext i1 %13 to i8
  %15 = getelementptr inbounds [1 x i8], ptr %3, i32 0, i32 0
  store i8 %14, ptr %15, align 1
  ret void
}

attributes #0 = { "nvvm.reqntid"="1,1,1" }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{}
