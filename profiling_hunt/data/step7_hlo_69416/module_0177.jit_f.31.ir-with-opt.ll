; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_compare_minimum_select_fusion(ptr noalias nofree readonly align 256 captures(none) dereferenceable(8) %0, ptr noalias nofree readonly align 256 captures(none) dereferenceable(8) %1, ptr noalias nofree readonly align 256 captures(none) dereferenceable(8) %2, ptr noalias nofree align 256 captures(none) dereferenceable(8) %3, ptr noalias nofree writeonly align 256 captures(none) dereferenceable(8) initializes((0, 8)) %4, ptr noalias nofree readnone align 256 captures(none) dereferenceable(8) %5, ptr noalias nofree writeonly align 256 captures(none) dereferenceable(1) initializes((0, 1)) %6) local_unnamed_addr #0 {
  %8 = addrspacecast ptr %3 to ptr addrspace(1)
  %9 = addrspacecast ptr %1 to ptr addrspace(1)
  %10 = addrspacecast ptr %2 to ptr addrspace(1)
  %11 = addrspacecast ptr %0 to ptr addrspace(1)
  %12 = addrspacecast ptr %4 to ptr addrspace(1)
  %13 = addrspacecast ptr %6 to ptr addrspace(1)
  %14 = load double, ptr addrspace(1) %8, align 256
  %15 = fmul double %14, 1.562500e-02
  %16 = fdiv double 1.000000e+00, %15
  %17 = tail call i32 @llvm.nvvm.d2i.hi(double %16) #5
  %18 = tail call i32 @llvm.nvvm.d2i.hi(double 2.000000e-01) #5
  %19 = and i32 %18, 2146435072
  %20 = tail call double @llvm.nvvm.fabs.f64(double %16)
  %21 = fcmp oeq double %16, 0.000000e+00
  br i1 %21, label %22, label %27

22:                                               ; preds = %7
  %23 = icmp eq i32 %19, 1126170624
  %thi.0.i = select i1 %23, i32 %17, i32 0
  %24 = icmp slt i32 %18, 0
  %25 = or i32 %thi.0.i, 2146435072
  %thi.1.i = select i1 %24, i32 %25, i32 %thi.0.i
  %26 = tail call double @llvm.nvvm.lohi.i2d(i32 0, i32 %thi.1.i) #5
  br label %30

27:                                               ; preds = %7
  %28 = icmp slt i32 %17, 0
  %29 = tail call fastcc double @__internal_accurate_pow(double %20)
  %t.1.i = select i1 %28, double -qnan, double %29
  br label %30

30:                                               ; preds = %27, %22
  %t.2.i = phi double [ %26, %22 ], [ %t.1.i, %27 ]
  %31 = fadd double %16, 2.000000e-01
  %32 = tail call i32 @llvm.nvvm.d2i.hi(double %31) #5
  %33 = and i32 %32, 2146435072
  %34 = icmp eq i32 %33, 2146435072
  br i1 %34, label %35, label %__nv_pow.exit

35:                                               ; preds = %30
  %36 = fcmp uno double %20, 0.000000e+00
  br i1 %36, label %37, label %39

37:                                               ; preds = %35
  %38 = tail call double @llvm.nvvm.add.rn.d(double %16, double 2.000000e-01) #5
  br label %__nv_pow.exit

39:                                               ; preds = %35
  %40 = fcmp oeq double %20, +inf
  br i1 %40, label %41, label %__nv_pow.exit

41:                                               ; preds = %39
  %42 = icmp slt i32 %17, 0
  %43 = icmp eq i32 %19, 1126170624
  %.inv.i = icmp slt i32 %18, 0
  %spec.select.i = select i1 %.inv.i, i32 0, i32 2146435072
  %44 = and i32 %18, 2147483647
  %45 = icmp ne i32 %44, 1071644672
  %46 = and i1 %43, %45
  %or.cond5.i = and i1 %46, %42
  %47 = or disjoint i32 %spec.select.i, -2147483648
  %thi.6.i = select i1 %or.cond5.i, i32 %47, i32 %spec.select.i
  %48 = tail call double @llvm.nvvm.lohi.i2d(i32 0, i32 %thi.6.i) #5
  br label %__nv_pow.exit

__nv_pow.exit:                                    ; preds = %30, %37, %39, %41
  %t.6.i = phi double [ %t.2.i, %30 ], [ %38, %37 ], [ %t.2.i, %39 ], [ %48, %41 ]
  %49 = fcmp oeq double %16, 1.000000e+00
  %50 = fcmp olt double %15, 1.000000e+00
  %51 = select i1 %50, double 1.000000e+00, double 2.000000e-01
  %52 = fmul double %t.6.i, 9.000000e-01
  %53 = select i1 %49, double 9.000000e-01, double %52
  %54 = load double, ptr addrspace(1) %9, align 256, !invariant.load !4
  %55 = load double, ptr addrspace(1) %10, align 256, !invariant.load !4
  %56 = select i1 %50, double %54, double %55
  %57 = tail call double @llvm.minimum.f64(double %56, double 1.000000e-01)
  %58 = fsub double 1.000000e-01, %57
  %59 = select i1 %50, double 5.000000e+00, double 9.000000e-01
  %60 = tail call nsz double @llvm.maximum.f64(double %51, double %53)
  %61 = fmul double %58, 5.000000e-01
  %62 = load double, ptr addrspace(1) %11, align 256, !invariant.load !4
  %63 = tail call nsz double @llvm.minimum.f64(double %59, double %60)
  %64 = zext i1 %50 to i8
  %65 = fadd double %57, %61
  %66 = fmul double %62, %63
  %67 = fadd double %56, %66
  %68 = fcmp ogt double %67, f0x3FB9999999999936
  %69 = select i1 %50, double 1.000000e-01, double %65
  %70 = select i1 %68, double %69, double %67
  store double %70, ptr addrspace(1) %12, align 256
  store double %57, ptr addrspace(1) %8, align 256
  store i8 %64, ptr addrspace(1) %13, align 256
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
