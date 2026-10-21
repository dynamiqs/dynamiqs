; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @loop_concatenate_fusion(ptr noalias align 256 dereferenceable(8) %0, ptr noalias align 256 dereferenceable(8) %1, ptr noalias align 256 dereferenceable(320) %2, ptr noalias align 256 dereferenceable(8) %3, ptr noalias align 256 dereferenceable(8) %4, ptr noalias align 256 dereferenceable(56) %5) #0 {
  %7 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !1
  %8 = sext i32 %7 to i64
  %9 = icmp ult i64 %8, 3
  br i1 %9, label %10, label %46

10:                                               ; preds = %6
  %11 = icmp ult i64 %8, 1
  br i1 %11, label %12, label %22

12:                                               ; preds = %10
  %13 = call double @fused_concatenate_div_22_3(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4)
  %14 = fmul double %13, f0x3FF6FB818F28C58D
  %15 = fmul double %13, f0xBFF0D9735318A381
  %16 = fadd double %13, f0xBFF547A290C89901
  %17 = fmul double %13, %13
  %18 = fsub double %17, %14
  %19 = fmul double %15, %16
  %20 = fadd double %18, f0x3FE6D8F021FA6FA2
  %21 = fmul double %19, %20
  br label %43

22:                                               ; preds = %10
  %23 = icmp ult i64 %8, 2
  br i1 %23, label %24, label %32

24:                                               ; preds = %22
  %25 = call double @fused_concatenate_div_22_3(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4)
  %26 = fmul double %25, f0x400192C0D16E8162
  %27 = fmul double %25, %25
  %28 = fsub double %27, %26
  %29 = fmul double %27, 1.017000e-01
  %30 = fadd double %28, f0x3FF4B842744B8427
  %31 = fmul double %29, %30
  br label %40

32:                                               ; preds = %22
  %33 = call double @fused_concatenate_div_22_3(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4)
  %34 = fmul double %33, f0x40031535C3D29ED4
  %35 = fmul double %33, %33
  %36 = fsub double %35, %34
  %37 = fmul double %35, f0x4003ECCDFF932DD6
  %38 = fadd double %36, f0x3FF93FA14B77DC7C
  %39 = fmul double %37, %38
  br label %40

40:                                               ; preds = %24, %32
  %41 = phi double [ %39, %32 ], [ %31, %24 ]
  br label %42

42:                                               ; preds = %40
  br label %43

43:                                               ; preds = %12, %42
  %44 = phi double [ %41, %42 ], [ %21, %12 ]
  br label %45

45:                                               ; preds = %43
  br label %93

46:                                               ; preds = %6
  %47 = icmp ult i64 %8, 5
  br i1 %47, label %48, label %69

48:                                               ; preds = %46
  %49 = icmp ult i64 %8, 4
  br i1 %49, label %50, label %58

50:                                               ; preds = %48
  %51 = call double @fused_concatenate_div_22_3(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4)
  %52 = fadd double %51, f0xBFF3795C8B658EFE
  %53 = fmul double %52, f0xC0308C5078902664
  %54 = fadd double %51, f0xBFE3B7F193DF8CAF
  %55 = fmul double %53, %54
  %56 = fmul double %51, %51
  %57 = fmul double %55, %56
  br label %66

58:                                               ; preds = %48
  %59 = call double @fused_concatenate_div_22_3(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4)
  %60 = fadd double %59, f0xBFF33FC7986B7EF9
  %61 = fmul double %60, f0x4047B0942CF93616
  %62 = fadd double %59, f0xBFE50EB9322562EF
  %63 = fmul double %61, %62
  %64 = fmul double %59, %59
  %65 = fmul double %63, %64
  br label %66

66:                                               ; preds = %50, %58
  %67 = phi double [ %65, %58 ], [ %57, %50 ]
  br label %68

68:                                               ; preds = %66
  br label %90

69:                                               ; preds = %46
  %70 = icmp ult i64 %8, 6
  br i1 %70, label %71, label %79

71:                                               ; preds = %69
  %72 = call double @fused_concatenate_div_22_3(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4)
  %73 = fadd double %72, -1.200000e+00
  %74 = fmul double %73, f0xC0416F71B780910E
  %75 = fadd double %72, f0xBFE5555555555555
  %76 = fmul double %74, %75
  %77 = fmul double %72, %72
  %78 = fmul double %76, %77
  br label %87

79:                                               ; preds = %69
  %80 = call double @fused_concatenate_div_22_3(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4)
  %81 = fadd double %80, -1.000000e+00
  %82 = fmul double %81, 2.500000e+00
  %83 = fadd double %80, -6.000000e-01
  %84 = fmul double %82, %83
  %85 = fmul double %80, %80
  %86 = fmul double %84, %85
  br label %87

87:                                               ; preds = %71, %79
  %88 = phi double [ %86, %79 ], [ %78, %71 ]
  br label %89

89:                                               ; preds = %87
  br label %90

90:                                               ; preds = %68, %89
  %91 = phi double [ %88, %89 ], [ %67, %68 ]
  br label %92

92:                                               ; preds = %90
  br label %93

93:                                               ; preds = %45, %92
  %94 = phi double [ %91, %92 ], [ %44, %45 ]
  br label %95

95:                                               ; preds = %93
  %96 = getelementptr inbounds [7 x double], ptr %5, i32 0, i64 %8
  store double %94, ptr %96, align 8
  ret void
}

define internal double @fused_concatenate_div_22_3(ptr noalias %0, ptr noalias %1, ptr noalias %2, ptr noalias %3, ptr noalias %4) {
  %6 = getelementptr inbounds [1 x i64], ptr %3, i32 0, i32 0
  %7 = load i64, ptr %6, align 4, !invariant.load !2
  %8 = getelementptr inbounds [1 x i64], ptr %4, i32 0, i32 0
  %9 = load i64, ptr %8, align 4, !invariant.load !2
  %10 = add i64 %7, %9
  %11 = icmp slt i64 %10, 0
  %12 = add i64 %10, 40
  %13 = select i1 %11, i64 %12, i64 %10
  %14 = call i64 @llvm.smin.i64(i64 %13, i64 39)
  %15 = call i64 @llvm.smax.i64(i64 %14, i64 0)
  %16 = getelementptr inbounds [40 x double], ptr %2, i32 0, i64 %15
  %17 = load double, ptr %16, align 8, !invariant.load !2
  %18 = getelementptr inbounds [1 x double], ptr %1, i32 0, i32 0
  %19 = load double, ptr %18, align 8, !invariant.load !2
  %20 = getelementptr inbounds [1 x double], ptr %0, i32 0, i32 0
  %21 = load double, ptr %20, align 8, !invariant.load !2
  %22 = fsub double %17, %21
  %23 = fcmp oeq double %21, %19
  %24 = fsub double %19, %21
  %25 = select i1 %23, double 0.000000e+00, double %22
  %26 = select i1 %23, double 1.000000e+00, double %24
  %27 = fdiv double %25, %26
  ret double %27
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.smin.i64(i64, i64) #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.smax.i64(i64, i64) #2

attributes #0 = { "nvvm.reqntid"="7,1,1" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 0, i32 7}
!2 = !{}
