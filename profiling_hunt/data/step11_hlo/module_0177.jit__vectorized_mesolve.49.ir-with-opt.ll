; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_compare_minimum_select_fusion(ptr noalias nofree readonly align 256 captures(none) dereferenceable(8) %0, ptr noalias nofree readonly align 256 captures(none) dereferenceable(8) %1, ptr noalias nofree readonly align 256 captures(none) dereferenceable(8) %2, ptr noalias nofree readonly align 256 captures(none) dereferenceable(8) %3, ptr noalias nofree readonly align 16 captures(none) dereferenceable(8) %4, ptr noalias nofree readonly align 16 captures(none) dereferenceable(8) %5, ptr noalias nofree readonly align 16 captures(none) dereferenceable(8) %6, ptr noalias nofree align 256 captures(none) dereferenceable(8) %7, ptr noalias nofree readnone align 256 captures(none) dereferenceable(8) %8, ptr noalias nofree writeonly align 256 captures(none) dereferenceable(8) initializes((0, 8)) %9, ptr noalias nofree writeonly align 256 captures(none) dereferenceable(1) initializes((0, 1)) %10) local_unnamed_addr #0 {
  %12 = addrspacecast ptr %7 to ptr addrspace(1)
  %13 = addrspacecast ptr %5 to ptr addrspace(1)
  %14 = addrspacecast ptr %6 to ptr addrspace(1)
  %15 = addrspacecast ptr %4 to ptr addrspace(1)
  %16 = addrspacecast ptr %0 to ptr addrspace(1)
  %17 = addrspacecast ptr %2 to ptr addrspace(1)
  %18 = addrspacecast ptr %3 to ptr addrspace(1)
  %19 = addrspacecast ptr %1 to ptr addrspace(1)
  %20 = addrspacecast ptr %9 to ptr addrspace(1)
  %21 = addrspacecast ptr %10 to ptr addrspace(1)
  %22 = load double, ptr addrspace(1) %12, align 256
  %23 = fmul double %22, 1.562500e-02
  %24 = fdiv double 1.000000e+00, %23
  %25 = tail call i32 @llvm.nvvm.d2i.hi(double %24) #5
  %26 = tail call i32 @llvm.nvvm.d2i.hi(double 2.000000e-01) #5
  %27 = and i32 %26, 2146435072
  %28 = tail call double @llvm.nvvm.fabs.f64(double %24)
  %29 = fcmp oeq double %24, 0.000000e+00
  br i1 %29, label %30, label %35

30:                                               ; preds = %11
  %31 = icmp eq i32 %27, 1126170624
  %thi.0.i = select i1 %31, i32 %25, i32 0
  %32 = icmp slt i32 %26, 0
  %33 = or i32 %thi.0.i, 2146435072
  %thi.1.i = select i1 %32, i32 %33, i32 %thi.0.i
  %34 = tail call double @llvm.nvvm.lohi.i2d(i32 0, i32 %thi.1.i) #5
  br label %38

35:                                               ; preds = %11
  %36 = icmp slt i32 %25, 0
  %37 = tail call fastcc double @__internal_accurate_pow(double %28)
  %t.1.i = select i1 %36, double -qnan, double %37
  br label %38

38:                                               ; preds = %35, %30
  %t.2.i = phi double [ %34, %30 ], [ %t.1.i, %35 ]
  %39 = fadd double %24, 2.000000e-01
  %40 = tail call i32 @llvm.nvvm.d2i.hi(double %39) #5
  %41 = and i32 %40, 2146435072
  %42 = icmp eq i32 %41, 2146435072
  br i1 %42, label %43, label %__nv_pow.exit

43:                                               ; preds = %38
  %44 = fcmp uno double %28, 0.000000e+00
  br i1 %44, label %45, label %47

45:                                               ; preds = %43
  %46 = tail call double @llvm.nvvm.add.rn.d(double %24, double 2.000000e-01) #5
  br label %__nv_pow.exit

47:                                               ; preds = %43
  %48 = fcmp oeq double %28, +inf
  br i1 %48, label %49, label %__nv_pow.exit

49:                                               ; preds = %47
  %50 = icmp slt i32 %25, 0
  %51 = icmp eq i32 %27, 1126170624
  %.inv.i = icmp slt i32 %26, 0
  %spec.select.i = select i1 %.inv.i, i32 0, i32 2146435072
  %52 = and i32 %26, 2147483647
  %53 = icmp ne i32 %52, 1071644672
  %54 = and i1 %51, %53
  %or.cond5.i = and i1 %54, %50
  %55 = or disjoint i32 %spec.select.i, -2147483648
  %thi.6.i = select i1 %or.cond5.i, i32 %55, i32 %spec.select.i
  %56 = tail call double @llvm.nvvm.lohi.i2d(i32 0, i32 %thi.6.i) #5
  br label %__nv_pow.exit

__nv_pow.exit:                                    ; preds = %38, %45, %47, %49
  %t.6.i = phi double [ %t.2.i, %38 ], [ %46, %45 ], [ %t.2.i, %47 ], [ %56, %49 ]
  %57 = load double, ptr addrspace(1) %13, align 16, !invariant.load !4
  %58 = fcmp oeq double %24, 1.000000e+00
  %t.7.i = select i1 %58, double 1.000000e+00, double %t.6.i
  %59 = load double, ptr addrspace(1) %14, align 16, !invariant.load !4
  %60 = load double, ptr addrspace(1) %15, align 16, !invariant.load !4
  %61 = fcmp olt double %23, 1.000000e+00
  %62 = select i1 %61, double 1.000000e+00, double %57
  %63 = fmul double %t.7.i, %60
  %64 = load double, ptr addrspace(1) %16, align 256, !invariant.load !4
  %65 = load double, ptr addrspace(1) %17, align 256, !invariant.load !4
  %66 = load double, ptr addrspace(1) %18, align 256, !invariant.load !4
  %67 = select i1 %61, double %65, double %66
  %68 = tail call double @llvm.minimum.f64(double %67, double %64)
  %69 = fsub double %64, %68
  %70 = select i1 %61, double %59, double %60
  %71 = tail call double @llvm.maximum.f64(double %62, double %63)
  %72 = fmul double %69, 5.000000e-01
  %73 = fsub double %65, %66
  %74 = tail call double @llvm.minimum.f64(double %70, double %71)
  %75 = load double, ptr addrspace(1) %19, align 256, !invariant.load !4
  %76 = zext i1 %61 to i8
  %77 = fadd double %68, %72
  %78 = fmul double %74, %73
  %79 = fadd double %67, %78
  %80 = fcmp ogt double %79, %75
  %81 = select i1 %61, double %64, double %77
  %82 = select i1 %80, double %81, double %79
  store double %82, ptr addrspace(1) %12, align 256
  store double %68, ptr addrspace(1) %20, align 256
  store i8 %76, ptr addrspace(1) %21, align 256
  ret void
}

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.minimum.f64(double, double) #1

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.maximum.f64(double, double) #1

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.nvvm.d2i.hi(double) #1

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.nvvm.fabs.f64(double) #1

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none)
define internal fastcc double @__internal_accurate_pow(double %a) unnamed_addr #2 {
__internal_log_ext_prec.exit:
  %0 = tail call i32 @llvm.nvvm.d2i.hi(double %a) #5
  %1 = tail call i32 @llvm.nvvm.d2i.lo(double %a) #5
  %2 = lshr i32 %0, 20
  %3 = icmp eq i32 %2, 0
  %4 = fmul double %a, f0x4350000000000000
  %5 = tail call i32 @llvm.nvvm.d2i.hi(double %4) #5
  %6 = tail call i32 @llvm.nvvm.d2i.lo(double %4) #5
  %7 = lshr i32 %5, 20
  %8 = add nsw i32 %7, -54
  %ilo.i.0 = select i1 %3, i32 %6, i32 %1
  %ihi.i.0 = select i1 %3, i32 %5, i32 %0
  %expo.i.0 = select i1 %3, i32 %8, i32 %2
  %9 = and i32 %ihi.i.0, -2146435073
  %10 = or disjoint i32 %9, 1072693248
  %11 = tail call double @llvm.nvvm.lohi.i2d(i32 %ilo.i.0, i32 %10) #5
  %12 = icmp ugt i32 %10, 1073127582
  %13 = tail call i32 @llvm.nvvm.d2i.lo(double %11) #5
  %14 = tail call i32 @llvm.nvvm.d2i.hi(double %11) #5
  %15 = add i32 %14, -1048576
  %16 = tail call double @llvm.nvvm.lohi.i2d(i32 %13, i32 %15) #5
  %m.i.0 = select i1 %12, double %16, double %11
  %expo.i.1.v = select i1 %12, i32 -1022, i32 -1023
  %expo.i.1 = add nsw i32 %expo.i.1.v, %expo.i.0
  %17 = fadd double %m.i.0, -1.000000e+00
  %18 = fadd double %m.i.0, 1.000000e+00
  %19 = tail call double @llvm.nvvm.rcp.approx.ftz.d(double %18) #5
  %20 = fneg double %18
  %21 = tail call double @llvm.fma.f64(double %20, double %19, double 1.000000e+00)
  %22 = tail call double @llvm.fma.f64(double %21, double %21, double %21)
  %23 = tail call double @llvm.fma.f64(double %22, double %19, double %19)
  %24 = fmul double %17, %23
  %25 = fadd double %24, %24
  %26 = fmul double %25, %25
  %27 = tail call double @llvm.fma.f64(double %26, double f0x3EB0F5FF7D2CAFE2, double f0x3ED0F5D241AD3B5A)
  %28 = tail call double @llvm.fma.f64(double %27, double %26, double f0x3EF3B20A75488A3F)
  %29 = tail call double @llvm.fma.f64(double %28, double %26, double f0x3F1745CDE4FAECD5)
  %30 = tail call double @llvm.fma.f64(double %29, double %26, double f0x3F3C71C7258A578B)
  %31 = tail call double @llvm.fma.f64(double %30, double %26, double f0x3F6249249242B910)
  %32 = tail call double @llvm.fma.f64(double %31, double %26, double f0x3F89999999999DFB)
  %33 = fmul double %26, %32
  %34 = fsub double %17, %25
  %35 = fmul double %34, 2.000000e+00
  %36 = fneg double %25
  %37 = tail call double @llvm.fma.f64(double %36, double %17, double %35)
  %38 = fmul double %23, %37
  %39 = fadd double %33, f0x3FB5555555555555
  %40 = fsub double f0x3FB5555555555555, %39
  %41 = fadd double %33, %40
  %42 = fadd double %41, f0xBC46A4CB00B9E7B0
  %43 = fadd double %39, %42
  %44 = fsub double %39, %43
  %45 = fadd double %42, %44
  %46 = tail call double @llvm.nvvm.mul.rn.d(double %25, double %25) #5
  %47 = fneg double %46
  %48 = tail call double @llvm.fma.f64(double %25, double %25, double %47)
  %49 = tail call i32 @llvm.nvvm.d2i.lo(double %38) #5
  %50 = tail call i32 @llvm.nvvm.d2i.hi(double %38) #5
  %51 = add i32 %50, 1048576
  %52 = tail call double @llvm.nvvm.lohi.i2d(i32 %49, i32 %51) #5
  %53 = tail call double @llvm.fma.f64(double %25, double %52, double %48)
  %54 = tail call double @llvm.nvvm.mul.rn.d(double %46, double %25) #5
  %55 = fneg double %54
  %56 = tail call double @llvm.fma.f64(double %46, double %25, double %55)
  %57 = tail call double @llvm.fma.f64(double %46, double %38, double %56)
  %58 = tail call double @llvm.fma.f64(double %53, double %25, double %57)
  %59 = tail call double @llvm.nvvm.mul.rn.d(double %43, double %54) #5
  %60 = fneg double %59
  %61 = tail call double @llvm.fma.f64(double %43, double %54, double %60)
  %62 = tail call double @llvm.fma.f64(double %43, double %58, double %61)
  %63 = tail call double @llvm.fma.f64(double %45, double %54, double %62)
  %64 = fadd double %59, %63
  %65 = fsub double %59, %64
  %66 = fadd double %63, %65
  %67 = fadd double %25, %64
  %68 = fsub double %25, %67
  %69 = fadd double %64, %68
  %70 = fadd double %66, %69
  %71 = fadd double %38, %70
  %72 = fadd double %67, %71
  %73 = fsub double %67, %72
  %74 = fadd double %71, %73
  %75 = xor i32 %expo.i.1, -2147483648
  %76 = tail call double @llvm.nvvm.lohi.i2d(i32 %75, i32 1127219200) #5
  %77 = tail call double @llvm.nvvm.lohi.i2d(i32 -2147483648, i32 1127219200) #5
  %78 = fsub double %76, %77
  %79 = tail call double @llvm.fma.f64(double %78, double f0x3FE62E42FEFA39EF, double %72)
  %80 = fneg double %78
  %81 = tail call double @llvm.fma.f64(double %80, double f0x3FE62E42FEFA39EF, double %79)
  %82 = fsub double %81, %72
  %83 = fsub double %74, %82
  %84 = tail call double @llvm.fma.f64(double %78, double f0x3C7ABC9E3B39803F, double %83)
  %85 = fadd double %79, %84
  %86 = fsub double %79, %85
  %87 = fadd double %84, %86
  %88 = tail call i32 @llvm.nvvm.d2i.hi(double 2.000000e-01) #5
  %89 = tail call i32 @llvm.nvvm.d2i.lo(double 2.000000e-01) #5
  %90 = shl i32 %88, 1
  %91 = icmp ugt i32 %90, -33554433
  %92 = and i32 %88, -15728641
  %spec.select = select i1 %91, i32 %92, i32 %88
  %93 = tail call double @llvm.nvvm.lohi.i2d(i32 %89, i32 %spec.select) #5
  %94 = tail call double @llvm.nvvm.mul.rn.d(double %85, double %93) #5
  %95 = fneg double %94
  %96 = tail call double @llvm.fma.f64(double %85, double %93, double %95)
  %97 = tail call double @llvm.fma.f64(double %87, double %93, double %96)
  %98 = fadd double %94, %97
  %99 = tail call double @llvm.fma.f64(double %98, double f0x3FF71547652B82FE, double f0x4338000000000000)
  %100 = tail call i32 @llvm.nvvm.d2i.lo(double %99) #5
  %101 = tail call double @llvm.nvvm.add.rn.d(double %99, double f0xC338000000000000) #5
  %102 = tail call double @llvm.fma.f64(double %101, double f0xBFE62E42FEFA39EF, double %98)
  %103 = tail call double @llvm.fma.f64(double %101, double f0xBC7ABC9E3B39803F, double %102)
  %104 = tail call double @llvm.fma.f64(double %103, double f0x3E5ADE1569CE2BDF, double f0x3E928AF3FCA213EA)
  %105 = tail call double @llvm.fma.f64(double %104, double %103, double f0x3EC71DEE62401315)
  %106 = tail call double @llvm.fma.f64(double %105, double %103, double f0x3EFA01997C89EB71)
  %107 = tail call double @llvm.fma.f64(double %106, double %103, double f0x3F2A01A014761F65)
  %108 = tail call double @llvm.fma.f64(double %107, double %103, double f0x3F56C16C1852B7AF)
  %109 = tail call double @llvm.fma.f64(double %108, double %103, double f0x3F81111111122322)
  %110 = tail call double @llvm.fma.f64(double %109, double %103, double f0x3FA55555555502A1)
  %111 = tail call double @llvm.fma.f64(double %110, double %103, double f0x3FC5555555555511)
  %112 = tail call double @llvm.fma.f64(double %111, double %103, double f0x3FE000000000000B)
  %113 = tail call double @llvm.fma.f64(double %112, double %103, double 1.000000e+00)
  %114 = tail call double @llvm.fma.f64(double %113, double %103, double 1.000000e+00)
  %115 = tail call i32 @llvm.nvvm.d2i.lo(double %114) #5
  %116 = tail call i32 @llvm.nvvm.d2i.hi(double %114) #5
  %117 = shl i32 %100, 20
  %118 = add i32 %116, %117
  %119 = tail call double @llvm.nvvm.lohi.i2d(i32 %115, i32 %118) #5
  %120 = tail call i32 @llvm.nvvm.d2i.hi(double %98) #5
  %121 = bitcast i32 %120 to float
  %122 = tail call float @llvm.nvvm.fabs.f32(float %121)
  %123 = fcmp olt float %122, f0x4086232B
  br i1 %123, label %__nv_exp.exit, label %__internal_fast_icmp_abs_lt.exit.i

__internal_fast_icmp_abs_lt.exit.i:               ; preds = %__internal_log_ext_prec.exit
  %124 = fcmp olt double %98, 0.000000e+00
  %125 = fadd double %98, +inf
  %z.i.0 = select i1 %124, double 0.000000e+00, double %125
  %126 = fcmp olt float %122, f0x40874800
  br i1 %126, label %127, label %__nv_exp.exit

127:                                              ; preds = %__internal_fast_icmp_abs_lt.exit.i
  %128 = sdiv i32 %100, 2
  %129 = shl i32 %128, 20
  %130 = add i32 %116, %129
  %131 = tail call double @llvm.nvvm.lohi.i2d(i32 %115, i32 %130) #5
  %132 = sub nsw i32 %100, %128
  %133 = shl i32 %132, 20
  %134 = add nsw i32 %133, 1072693248
  %135 = tail call double @llvm.nvvm.lohi.i2d(i32 0, i32 %134) #5
  %136 = fmul double %135, %131
  br label %__nv_exp.exit

__nv_exp.exit:                                    ; preds = %__internal_fast_icmp_abs_lt.exit.i, %127, %__internal_log_ext_prec.exit
  %z.i.2 = phi double [ %119, %__internal_log_ext_prec.exit ], [ %136, %127 ], [ %z.i.0, %__internal_fast_icmp_abs_lt.exit.i ]
  %137 = fsub double %94, %98
  %138 = fadd double %97, %137
  %139 = tail call double @llvm.nvvm.fabs.f64(double %z.i.2)
  %140 = fcmp oeq double %139, +inf
  %141 = tail call double @llvm.fma.f64(double %z.i.2, double %138, double %z.i.2)
  %spec.select1 = select i1 %140, double %z.i.2, double %141
  ret double %spec.select1
}

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.nvvm.lohi.i2d(i32, i32) #1

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.nvvm.add.rn.d(double, double) #1

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.nvvm.d2i.lo(double) #1

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare double @llvm.nvvm.rcp.approx.ftz.d(double) #3

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.nvvm.mul.rn.d(double, double) #1

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.nvvm.fabs.f32(float) #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fma.f64(double, double, double) #4

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "nvvm.reqntid"="1,1,1" }
attributes #1 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #4 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #5 = { nounwind }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}
!nvvmir.version = !{!3}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}
!3 = !{i32 2, i32 0}
!4 = !{}
