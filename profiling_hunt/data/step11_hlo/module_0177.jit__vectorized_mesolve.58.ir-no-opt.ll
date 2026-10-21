; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @loop_divide_fusion(ptr noalias align 256 dereferenceable(8) %0, ptr noalias align 256 dereferenceable(8) %1, ptr noalias align 256 dereferenceable(8) %2, ptr noalias align 256 dereferenceable(8) %3) #0 {
  %5 = getelementptr inbounds [1 x double], ptr %2, i32 0, i32 0
  %6 = load double, ptr %5, align 8, !invariant.load !1
  %7 = getelementptr inbounds [1 x double], ptr %0, i32 0, i32 0
  %8 = load double, ptr %7, align 8, !invariant.load !1
  %9 = getelementptr inbounds [1 x double], ptr %1, i32 0, i32 0
  %10 = load double, ptr %9, align 8, !invariant.load !1
  %11 = fsub double %6, %10
  %12 = fcmp oeq double %10, %8
  %13 = fsub double %8, %10
  %14 = select i1 %12, double 0.000000e+00, double %11
  %15 = select i1 %12, double 1.000000e+00, double %13
  %16 = fdiv double %14, %15
  %17 = getelementptr inbounds [1 x double], ptr %3, i32 0, i32 0
  store double %16, ptr %17, align 8
  ret void
}

attributes #0 = { "nvvm.reqntid"="1,1,1" }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{}
