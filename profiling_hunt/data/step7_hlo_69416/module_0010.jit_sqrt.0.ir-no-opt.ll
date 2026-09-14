; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

declare double @__nv_fabs(double)

declare double @__nv_sqrt(double)

define ptx_kernel void @loop_complex_fusion(ptr noalias align 16 dereferenceable(1008) %0, ptr noalias align 256 dereferenceable(1008) %1) #0 {
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !1
  %4 = getelementptr inbounds [63 x { double, double }], ptr %0, i32 0, i32 %3
  %5 = load { double, double }, ptr %4, align 8, !invariant.load !2
  %6 = extractvalue { double, double } %5, 0
  %7 = extractvalue { double, double } %5, 1
  %8 = call double @__nv_fabs(double %6)
  %9 = call double @__nv_fabs(double %7)
  %10 = fcmp olt double %8, %9
  %11 = fdiv double %8, %9
  %12 = fdiv double %9, %8
  %13 = fcmp oeq double %8, %9
  %14 = select i1 %10, double %11, double %12
  %15 = select i1 %13, double 1.000000e+00, double %14
  %16 = call double @__nv_fabs(double %15)
  %17 = call double @llvm.minimum.f64(double %16, double 1.000000e+00)
  %18 = call double @llvm.maximum.f64(double %16, double 1.000000e+00)
  %19 = fdiv double %17, %18
  %20 = fmul double %19, %19
  %21 = fmul double %18, %20
  %22 = fadd double %20, 1.000000e+00
  %23 = call double @llvm.minimum.f64(double %8, double %9)
  %24 = call double @llvm.maximum.f64(double %8, double %9)
  %25 = fdiv double %23, %24
  %26 = call double @__nv_sqrt(double %22)
  %27 = fcmp oeq double %26, 1.000000e+00
  %28 = fcmp ogt double %20, 0.000000e+00
  %29 = fmul double %21, 5.000000e-01
  %30 = fmul double %25, %25
  %31 = fmul double %24, %30
  %32 = fadd double %30, 1.000000e+00
  %33 = and i1 %27, %28
  %34 = fadd double %18, %29
  %35 = fmul double %18, %26
  %36 = call double @__nv_sqrt(double %32)
  %37 = fcmp oeq double %36, 1.000000e+00
  %38 = fcmp ogt double %30, 0.000000e+00
  %39 = fmul double %31, 5.000000e-01
  %40 = fcmp oeq double %18, %17
  %41 = fmul double %18, f0x3FF6A09E667F3BCD
  %42 = select i1 %33, double %34, double %35
  %43 = and i1 %37, %38
  %44 = fadd double %24, %39
  %45 = fmul double %24, %36
  %46 = call double @__nv_sqrt(double %8)
  %47 = call double @__nv_sqrt(double %9)
  %48 = fdiv double %46, %47
  %49 = fdiv double %47, %46
  %50 = select i1 %40, double %41, double %42
  %51 = fcmp oeq double %24, %23
  %52 = fmul double %24, f0x3FF6A09E667F3BCD
  %53 = select i1 %43, double %44, double %45
  %54 = select i1 %10, double %48, double %49
  %55 = fadd double %50, 1.000000e+00
  %56 = fadd double %15, %50
  %57 = select i1 %51, double %52, double %53
  %58 = select i1 %13, double 1.000000e+00, double %54
  %59 = call double @__nv_sqrt(double %55)
  %60 = call double @__nv_sqrt(double %56)
  %61 = fmul double %59, f0x3FE6A09E667F3BCC
  %62 = fmul double %60, f0x3FE6A09E667F3BCC
  %63 = fadd double %57, %8
  %64 = fmul double %47, %58
  %65 = fmul double %59, f0x3FF6A09E667F3BCD
  %66 = fmul double %60, f0x3FF6A09E667F3BCD
  %67 = fmul double %46, %61
  %68 = fmul double %47, %62
  %69 = fmul double %63, 5.000000e-01
  %70 = call double @__nv_sqrt(double %69)
  %71 = fcmp oeq double %70, 0.000000e+00
  %72 = fcmp oeq double %70, +inf
  %73 = fcmp ogt double %8, %9
  %74 = fdiv double %64, %65
  %75 = fdiv double %47, %66
  %76 = fmul double %70, 2.000000e+00
  %77 = select i1 %73, double %67, double %68
  %78 = or i1 %71, %72
  %79 = select i1 %73, double %74, double %75
  %80 = fdiv double %9, %76
  %81 = fmul double %46, f0x3FF19435CAFFA9F8
  %82 = select i1 %78, double %77, double %70
  %83 = fmul double %47, f0x3FDD203138F6C828
  %84 = select i1 %78, double %79, double %80
  %85 = select i1 %13, double %81, double %82
  %86 = fneg double %85
  %87 = fcmp olt double %7, 0.000000e+00
  %88 = select i1 %13, double %83, double %84
  %89 = fneg double %88
  %90 = fcmp oge double %6, 0.000000e+00
  %91 = fcmp olt double %6, 0.000000e+00
  %92 = select i1 %87, double %86, double %85
  %93 = select i1 %87, double %89, double %88
  %94 = select i1 %90, double %85, double %88
  %95 = select i1 %91, double %92, double %93
  %96 = insertvalue { double, double } poison, double %94, 0
  %97 = insertvalue { double, double } %96, double %95, 1
  %98 = getelementptr inbounds [63 x { double, double }], ptr %1, i32 0, i32 %3
  store { double, double } %97, ptr %98, align 8
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.minimum.f64(double, double) #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.maximum.f64(double, double) #2

attributes #0 = { "nvvm.reqntid"="63,1,1" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 0, i32 63}
!2 = !{}
