; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

declare double @__nv_pow(double, double)

define ptx_kernel void @loop_compare_minimum_select_fusion(ptr noalias align 256 dereferenceable(8) %0, ptr noalias align 256 dereferenceable(8) %1, ptr noalias align 256 dereferenceable(8) %2, ptr noalias align 256 dereferenceable(8) %3, ptr noalias align 16 dereferenceable(8) %4, ptr noalias align 16 dereferenceable(8) %5, ptr noalias align 16 dereferenceable(8) %6, ptr noalias align 256 dereferenceable(8) %7, ptr noalias align 256 dereferenceable(8) %8, ptr noalias align 256 dereferenceable(8) %9, ptr noalias align 256 dereferenceable(1) %10) #0 {
  %12 = getelementptr inbounds [1 x double], ptr %7, i32 0, i32 0
  %13 = load double, ptr %12, align 8
  %14 = fmul double %13, 1.562500e-02
  %15 = fdiv double 1.000000e+00, %14
  %16 = getelementptr inbounds [1 x double], ptr %5, i32 0, i32 0
  %17 = load double, ptr %16, align 8, !invariant.load !1
  %18 = call double @__nv_pow(double %15, double 2.000000e-01)
  %19 = getelementptr inbounds [1 x double], ptr %6, i32 0, i32 0
  %20 = load double, ptr %19, align 8, !invariant.load !1
  %21 = getelementptr inbounds [1 x double], ptr %4, i32 0, i32 0
  %22 = load double, ptr %21, align 8, !invariant.load !1
  %23 = fcmp olt double %14, 1.000000e+00
  %24 = select i1 %23, double 1.000000e+00, double %17
  %25 = fmul double %22, %18
  %26 = getelementptr inbounds [1 x double], ptr %0, i32 0, i32 0
  %27 = load double, ptr %26, align 8, !invariant.load !1
  %28 = getelementptr inbounds [1 x double], ptr %2, i32 0, i32 0
  %29 = load double, ptr %28, align 8, !invariant.load !1
  %30 = getelementptr inbounds [1 x double], ptr %3, i32 0, i32 0
  %31 = load double, ptr %30, align 8, !invariant.load !1
  %32 = select i1 %23, double %29, double %31
  %33 = call double @llvm.minimum.f64(double %32, double %27)
  %34 = fsub double %27, %33
  %35 = select i1 %23, double %20, double %22
  %36 = call double @llvm.maximum.f64(double %24, double %25)
  %37 = fmul double %34, 5.000000e-01
  %38 = fsub double %29, %31
  %39 = call double @llvm.minimum.f64(double %35, double %36)
  %40 = getelementptr inbounds [1 x double], ptr %1, i32 0, i32 0
  %41 = load double, ptr %40, align 8, !invariant.load !1
  %42 = zext i1 %23 to i8
  %43 = fadd double %33, %37
  %44 = fmul double %38, %39
  %45 = fadd double %32, %44
  %46 = fcmp ogt double %45, %41
  %47 = select i1 %23, double %27, double %43
  %48 = select i1 %46, double %47, double %45
  store double %48, ptr %12, align 8
  %49 = getelementptr inbounds [1 x double], ptr %9, i32 0, i32 0
  store double %33, ptr %49, align 8
  %50 = getelementptr inbounds [1 x i8], ptr %10, i32 0, i32 0
  store i8 %42, ptr %50, align 1
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
