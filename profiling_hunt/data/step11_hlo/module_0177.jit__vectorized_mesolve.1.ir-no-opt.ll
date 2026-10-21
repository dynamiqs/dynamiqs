; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @loop_minimum_multiply_select_fusion(ptr noalias align 16 dereferenceable(320) %0, ptr noalias align 256 dereferenceable(8) %1, ptr noalias align 256 dereferenceable(8) %2, ptr noalias align 256 dereferenceable(8) %3, ptr noalias align 256 dereferenceable(8) %4) #0 {
  %6 = getelementptr inbounds [40 x double], ptr %0, i32 0, i32 0
  %7 = load double, ptr %6, align 8, !invariant.load !1
  %8 = getelementptr inbounds [40 x double], ptr %0, i32 0, i32 39
  %9 = load double, ptr %8, align 8, !invariant.load !1
  %10 = fcmp olt double %7, %9
  %11 = select i1 %10, i64 1, i64 -1
  %12 = sitofp i64 %11 to double
  %13 = fmul double %7, %12
  %14 = fadd double %13, 1.000000e-02
  %15 = fmul double %9, %12
  %16 = call double @llvm.minimum.f64(double %14, double %15)
  %17 = getelementptr inbounds [1 x double], ptr %1, i32 0, i32 0
  store double %16, ptr %17, align 8
  %18 = getelementptr inbounds [1 x double], ptr %2, i32 0, i32 0
  store double %15, ptr %18, align 8
  %19 = getelementptr inbounds [1 x double], ptr %3, i32 0, i32 0
  store double %13, ptr %19, align 8
  %20 = getelementptr inbounds [1 x i64], ptr %4, i32 0, i32 0
  store i64 %11, ptr %20, align 4
  ret void
}

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.minimum.f64(double, double) #1

attributes #0 = { "nvvm.reqntid"="1,1,1" }
attributes #1 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{}
