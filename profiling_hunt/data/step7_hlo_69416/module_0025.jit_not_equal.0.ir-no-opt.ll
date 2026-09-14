; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @loop_compare_fusion(ptr noalias align 16 dereferenceable(16) %0, ptr noalias align 16 dereferenceable(8) %1, ptr noalias align 256 dereferenceable(1) %2) #0 {
  %4 = getelementptr inbounds [1 x i64], ptr %1, i32 0, i32 0
  %5 = load i64, ptr %4, align 4, !invariant.load !1
  %6 = getelementptr inbounds [1 x { double, double }], ptr %0, i32 0, i32 0
  %7 = load { double, double }, ptr %6, align 8, !invariant.load !1
  %8 = sitofp i64 %5 to double
  %9 = extractvalue { double, double } %7, 0
  %10 = extractvalue { double, double } %7, 1
  %11 = fcmp une double %9, %8
  %12 = fcmp une double %10, 0.000000e+00
  %13 = or i1 %11, %12
  %14 = zext i1 %13 to i8
  %15 = getelementptr inbounds [1 x i8], ptr %2, i32 0, i32 0
  store i8 %14, ptr %15, align 1
  ret void
}

attributes #0 = { "nvvm.reqntid"="1,1,1" }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{}
