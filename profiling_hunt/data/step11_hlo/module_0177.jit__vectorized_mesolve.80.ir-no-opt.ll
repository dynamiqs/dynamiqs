; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

declare double @__nv_fabs(double)

declare double @__nv_sqrt(double)

declare double @__nv_copysign(double, double)

define ptx_kernel void @loop_multiply_fusion_1(ptr noalias align 256 dereferenceable(32768) %0, ptr noalias align 256 dereferenceable(65536) %1, ptr noalias align 256 dereferenceable(65536) %2, ptr noalias align 256 dereferenceable(65536) %3, ptr noalias align 16 dereferenceable(8) %4, ptr noalias align 16 dereferenceable(8) %5, ptr noalias align 256 dereferenceable(1) %6, ptr noalias align 256 dereferenceable(65536) %7) #0 {
  %9 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !1
  %10 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %11 = getelementptr inbounds [1 x i8], ptr %6, i32 0, i32 0
  %12 = load i8, ptr %11, align 1, !invariant.load !3
  %13 = mul i32 %9, 128
  %14 = add i32 %13, %10
  %15 = getelementptr inbounds [4096 x { double, double }], ptr %1, i32 0, i32 %14
  %16 = load { double, double }, ptr %15, align 8
  %17 = getelementptr inbounds [4096 x { double, double }], ptr %2, i32 0, i32 %14
  %18 = load { double, double }, ptr %17, align 8, !invariant.load !3
  %19 = trunc i8 %12 to i1
  %20 = select i1 %19, { double, double } %16, { double, double } %18
  %21 = getelementptr inbounds [4096 x double], ptr %0, i32 0, i32 %14
  %22 = load double, ptr %21, align 8, !invariant.load !3
  %23 = extractvalue { double, double } %20, 0
  %24 = extractvalue { double, double } %20, 1
  %25 = call double @__nv_fabs(double %23)
  %26 = call double @__nv_fabs(double %24)
  %27 = call double @llvm.maximum.f64(double %25, double %26)
  %28 = call double @llvm.minimum.f64(double %25, double %26)
  %29 = fdiv double %28, %27
  %30 = fmul double %29, %29
  %31 = fadd double %30, 1.000000e+00
  %32 = call double @__nv_sqrt(double %31)
  %33 = fmul double %27, %32
  %34 = fcmp uno double %33, %33
  %35 = select i1 %34, double %28, double %33
  %36 = call double @llvm.maximum.f64(double %22, double %35)
  %37 = getelementptr inbounds [1 x double], ptr %4, i32 0, i32 0
  %38 = load double, ptr %37, align 8, !invariant.load !3
  %39 = getelementptr inbounds [1 x double], ptr %5, i32 0, i32 0
  %40 = load double, ptr %39, align 8, !invariant.load !3
  %41 = fmul double %36, %38
  %42 = getelementptr inbounds [4096 x { double, double }], ptr %3, i32 0, i32 %14
  %43 = load { double, double }, ptr %42, align 8, !invariant.load !3
  %44 = extractvalue { double, double } %43, 0
  %45 = extractvalue { double, double } %43, 1
  %46 = fcmp une double %44, %44
  %47 = fcmp une double %45, %45
  %48 = or i1 %46, %47
  %49 = fadd double %40, %41
  %50 = select i1 %48, { double, double } { double +inf, double 0.000000e+00 }, { double, double } %43
  %51 = extractvalue { double, double } %50, 0
  %52 = extractvalue { double, double } %50, 1
  %53 = fdiv double %49, 0.000000e+00
  %54 = fmul double %53, %49
  %55 = fadd double %54, 0.000000e+00
  %56 = fmul double %51, %53
  %57 = fadd double %56, %52
  %58 = fdiv double %57, %55
  %59 = fmul double %52, %53
  %60 = fsub double %59, %51
  %61 = fdiv double %60, %55
  %62 = fdiv double 0.000000e+00, %49
  %63 = fmul double %62, 0.000000e+00
  %64 = fadd double %49, %63
  %65 = fmul double %52, %62
  %66 = fadd double %51, %65
  %67 = fdiv double %66, %64
  %68 = fmul double %51, %62
  %69 = fsub double %52, %68
  %70 = fdiv double %69, %64
  %71 = call double @__nv_fabs(double %49)
  %72 = fcmp oeq double %71, 0.000000e+00
  %73 = fcmp ord double %51, 0.000000e+00
  %74 = fcmp ord double %52, 0.000000e+00
  %75 = or i1 %73, %74
  %76 = and i1 %75, %72
  %77 = call double @__nv_copysign(double +inf, double %49)
  %78 = fmul double %77, %51
  %79 = fmul double %77, %52
  %80 = fcmp one double %71, +inf
  %81 = call double @__nv_fabs(double %51)
  %82 = fcmp oeq double %81, +inf
  %83 = call double @__nv_fabs(double %52)
  %84 = fcmp oeq double %83, +inf
  %85 = or i1 %82, %84
  %86 = and i1 %85, %80
  %87 = select i1 %82, double 1.000000e+00, double 0.000000e+00
  %88 = call double @__nv_copysign(double %87, double %51)
  %89 = select i1 %84, double 1.000000e+00, double 0.000000e+00
  %90 = call double @__nv_copysign(double %89, double %52)
  %91 = fmul double %88, %49
  %92 = fmul double %90, 0.000000e+00
  %93 = fadd double %91, %92
  %94 = fmul double %93, +inf
  %95 = fmul double %88, 0.000000e+00
  %96 = fmul double %90, %49
  %97 = fsub double %96, %95
  %98 = fmul double %97, +inf
  %99 = fcmp one double %81, +inf
  %100 = fcmp one double %83, +inf
  %101 = and i1 %99, %100
  %102 = fcmp oeq double %71, +inf
  %103 = and i1 %101, %102
  %104 = select i1 %102, double 1.000000e+00, double 0.000000e+00
  %105 = call double @__nv_copysign(double %104, double %49)
  %106 = fmul double %51, %105
  %107 = fmul double %52, 0.000000e+00
  %108 = fadd double %106, %107
  %109 = fmul double %108, 0.000000e+00
  %110 = fmul double %52, %105
  %111 = fmul double %51, 0.000000e+00
  %112 = fsub double %110, %111
  %113 = fmul double %112, 0.000000e+00
  %114 = fcmp olt double %71, 0.000000e+00
  %115 = select i1 %114, double %58, double %67
  %116 = select i1 %114, double %61, double %70
  %117 = select i1 %103, double %109, double %115
  %118 = select i1 %103, double %113, double %116
  %119 = select i1 %86, double %94, double %117
  %120 = select i1 %86, double %98, double %118
  %121 = select i1 %76, double %78, double %119
  %122 = select i1 %76, double %79, double %120
  %123 = fcmp uno double %115, 0.000000e+00
  %124 = fcmp uno double %116, 0.000000e+00
  %125 = and i1 %123, %124
  %126 = select i1 %125, double %121, double %115
  %127 = select i1 %125, double %122, double %116
  %128 = fneg double %127
  %129 = fmul double %126, %126
  %130 = fmul double %128, %127
  %131 = fmul double %128, %126
  %132 = fmul double %126, %127
  %133 = fsub double %129, %130
  %134 = fadd double %131, %132
  %135 = insertvalue { double, double } poison, double %133, 0
  %136 = insertvalue { double, double } %135, double %134, 1
  store { double, double } %136, ptr %15, align 8
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.maximum.f64(double, double) #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.minimum.f64(double, double) #2

attributes #0 = { "nvvm.reqntid"="128,1,1" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 0, i32 32}
!2 = !{i32 0, i32 128}
!3 = !{}
