; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

declare double @__nv_pow(double, double)

define ptx_kernel void @loop_compare_minimum_select_fusion(ptr noalias align 256 dereferenceable(8) %0, ptr noalias align 256 dereferenceable(8) %1, ptr noalias align 256 dereferenceable(8) %2, ptr noalias align 256 dereferenceable(8) %3, ptr noalias align 256 dereferenceable(8) %4, ptr noalias align 256 dereferenceable(8) %5, ptr noalias align 256 dereferenceable(1) %6) #0 {
  %8 = getelementptr inbounds [1 x double], ptr %3, i32 0, i32 0
  %9 = load double, ptr %8, align 8
  %10 = fmul double %9, 1.562500e-02
  %11 = fdiv double 1.000000e+00, %10
  %12 = call double @__nv_pow(double %11, double 2.000000e-01)
  %13 = fcmp olt double %10, 1.000000e+00
  %14 = select i1 %13, double 1.000000e+00, double 2.000000e-01
  %15 = fmul double %12, 9.000000e-01
  %16 = getelementptr inbounds [1 x double], ptr %1, i32 0, i32 0
  %17 = load double, ptr %16, align 8, !invariant.load !1
  %18 = getelementptr inbounds [1 x double], ptr %2, i32 0, i32 0
  %19 = load double, ptr %18, align 8
  %20 = select i1 %13, double %17, double %19
  %21 = call double @llvm.minimum.f64(double %20, double 1.000000e-01)
  %22 = fsub double 1.000000e-01, %21
  %23 = select i1 %13, double 5.000000e+00, double 9.000000e-01
  %24 = call double @llvm.maximum.f64(double %14, double %15)
  %25 = fmul double %22, 5.000000e-01
  %26 = getelementptr inbounds [1 x double], ptr %0, i32 0, i32 0
  %27 = load double, ptr %26, align 8, !invariant.load !1
  %28 = call double @llvm.minimum.f64(double %23, double %24)
  %29 = zext i1 %13 to i8
  %30 = fadd double %21, %25
  %31 = fmul double %27, %28
  %32 = fadd double %20, %31
  %33 = fcmp ogt double %32, f0x3FB9999999999936
  %34 = select i1 %13, double 1.000000e-01, double %30
  %35 = select i1 %33, double %34, double %32
  store double %35, ptr %8, align 8
  store double %21, ptr %18, align 8
  %36 = getelementptr inbounds [1 x i8], ptr %6, i32 0, i32 0
  store i8 %29, ptr %36, align 1
  ret void
}

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.minimum.f64(double, double) #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.maximum.f64(double, double) #1

attributes #0 = { "nvvm.reqntid"="1,1,1" }
attributes #1 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{}
