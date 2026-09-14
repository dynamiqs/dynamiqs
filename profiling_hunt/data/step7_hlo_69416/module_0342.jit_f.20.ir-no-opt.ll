; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

declare double @__nv_fabs(double)

declare double @__nv_sqrt(double)

declare double @__nv_copysign(double, double)

define ptx_kernel void @loop_multiply_fusion(ptr noalias align 256 dereferenceable(32768) %0, ptr noalias align 256 dereferenceable(65536) %1, ptr noalias align 256 dereferenceable(65536) %2, ptr noalias align 256 dereferenceable(65536) %3, ptr noalias align 256 dereferenceable(1) %4, ptr noalias align 256 dereferenceable(65536) %5) #0 {
  %7 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !1
  %8 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %9 = getelementptr inbounds [1 x i8], ptr %4, i32 0, i32 0
  %10 = load i8, ptr %9, align 1, !invariant.load !3
  %11 = mul i32 %7, 128
  %12 = add i32 %11, %8
  %13 = getelementptr inbounds [4096 x { double, double }], ptr %1, i32 0, i32 %12
  %14 = load { double, double }, ptr %13, align 8
  %15 = getelementptr inbounds [4096 x { double, double }], ptr %2, i32 0, i32 %12
  %16 = load { double, double }, ptr %15, align 8, !invariant.load !3
  %17 = trunc i8 %10 to i1
  %18 = select i1 %17, { double, double } %14, { double, double } %16
  %19 = getelementptr inbounds [4096 x double], ptr %0, i32 0, i32 %12
  %20 = load double, ptr %19, align 8, !invariant.load !3
  %21 = extractvalue { double, double } %18, 0
  %22 = extractvalue { double, double } %18, 1
  %23 = call double @__nv_fabs(double %21)
  %24 = call double @__nv_fabs(double %22)
  %25 = call double @llvm.maximum.f64(double %23, double %24)
  %26 = call double @llvm.minimum.f64(double %23, double %24)
  %27 = fdiv double %26, %25
  %28 = fmul double %27, %27
  %29 = fadd double %28, 1.000000e+00
  %30 = call double @__nv_sqrt(double %29)
  %31 = fmul double %25, %30
  %32 = fcmp uno double %31, %31
  %33 = select i1 %32, double %26, double %31
  %34 = call double @llvm.maximum.f64(double %20, double %33)
  %35 = fmul double %34, f0x3EB0C6F7A0B5ED8D
  %36 = getelementptr inbounds [4096 x { double, double }], ptr %3, i32 0, i32 %12
  %37 = load { double, double }, ptr %36, align 8, !invariant.load !3
  %38 = extractvalue { double, double } %37, 0
  %39 = extractvalue { double, double } %37, 1
  %40 = fcmp une double %38, %38
  %41 = fcmp une double %39, %39
  %42 = or i1 %40, %41
  %43 = fadd double %35, f0x3EB0C6F7A0B5ED8D
  %44 = select i1 %42, { double, double } { double +inf, double 0.000000e+00 }, { double, double } %37
  %45 = extractvalue { double, double } %44, 0
  %46 = extractvalue { double, double } %44, 1
  %47 = fdiv double %43, 0.000000e+00
  %48 = fmul double %47, %43
  %49 = fadd double %48, 0.000000e+00
  %50 = fmul double %45, %47
  %51 = fadd double %50, %46
  %52 = fdiv double %51, %49
  %53 = fmul double %46, %47
  %54 = fsub double %53, %45
  %55 = fdiv double %54, %49
  %56 = fdiv double 0.000000e+00, %43
  %57 = fmul double %56, 0.000000e+00
  %58 = fadd double %43, %57
  %59 = fmul double %46, %56
  %60 = fadd double %45, %59
  %61 = fdiv double %60, %58
  %62 = fmul double %45, %56
  %63 = fsub double %46, %62
  %64 = fdiv double %63, %58
  %65 = call double @__nv_fabs(double %43)
  %66 = fcmp oeq double %65, 0.000000e+00
  %67 = fcmp ord double %45, 0.000000e+00
  %68 = fcmp ord double %46, 0.000000e+00
  %69 = or i1 %67, %68
  %70 = and i1 %69, %66
  %71 = call double @__nv_copysign(double +inf, double %43)
  %72 = fmul double %71, %45
  %73 = fmul double %71, %46
  %74 = fcmp one double %65, +inf
  %75 = call double @__nv_fabs(double %45)
  %76 = fcmp oeq double %75, +inf
  %77 = call double @__nv_fabs(double %46)
  %78 = fcmp oeq double %77, +inf
  %79 = or i1 %76, %78
  %80 = and i1 %79, %74
  %81 = select i1 %76, double 1.000000e+00, double 0.000000e+00
  %82 = call double @__nv_copysign(double %81, double %45)
  %83 = select i1 %78, double 1.000000e+00, double 0.000000e+00
  %84 = call double @__nv_copysign(double %83, double %46)
  %85 = fmul double %82, %43
  %86 = fmul double %84, 0.000000e+00
  %87 = fadd double %85, %86
  %88 = fmul double %87, +inf
  %89 = fmul double %82, 0.000000e+00
  %90 = fmul double %84, %43
  %91 = fsub double %90, %89
  %92 = fmul double %91, +inf
  %93 = fcmp one double %75, +inf
  %94 = fcmp one double %77, +inf
  %95 = and i1 %93, %94
  %96 = fcmp oeq double %65, +inf
  %97 = and i1 %95, %96
  %98 = select i1 %96, double 1.000000e+00, double 0.000000e+00
  %99 = call double @__nv_copysign(double %98, double %43)
  %100 = fmul double %45, %99
  %101 = fmul double %46, 0.000000e+00
  %102 = fadd double %100, %101
  %103 = fmul double %102, 0.000000e+00
  %104 = fmul double %46, %99
  %105 = fmul double %45, 0.000000e+00
  %106 = fsub double %104, %105
  %107 = fmul double %106, 0.000000e+00
  %108 = fcmp olt double %65, 0.000000e+00
  %109 = select i1 %108, double %52, double %61
  %110 = select i1 %108, double %55, double %64
  %111 = select i1 %97, double %103, double %109
  %112 = select i1 %97, double %107, double %110
  %113 = select i1 %80, double %88, double %111
  %114 = select i1 %80, double %92, double %112
  %115 = select i1 %70, double %72, double %113
  %116 = select i1 %70, double %73, double %114
  %117 = fcmp uno double %109, 0.000000e+00
  %118 = fcmp uno double %110, 0.000000e+00
  %119 = and i1 %117, %118
  %120 = select i1 %119, double %115, double %109
  %121 = select i1 %119, double %116, double %110
  %122 = fneg double %121
  %123 = fmul double %120, %120
  %124 = fmul double %122, %121
  %125 = fmul double %122, %120
  %126 = fmul double %120, %121
  %127 = fsub double %123, %124
  %128 = fadd double %125, %126
  %129 = insertvalue { double, double } poison, double %127, 0
  %130 = insertvalue { double, double } %129, double %128, 1
  store { double, double } %130, ptr %13, align 8
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
