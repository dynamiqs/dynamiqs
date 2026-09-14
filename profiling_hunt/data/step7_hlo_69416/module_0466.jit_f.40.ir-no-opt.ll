; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @input_add_fusion(ptr noalias align 256 dereferenceable(65536) %0, ptr noalias align 256 dereferenceable(65536) %1, ptr noalias align 256 dereferenceable(1008) %2, ptr noalias align 256 dereferenceable(1008) %3, ptr noalias align 256 dereferenceable(992) %4, ptr noalias align 256 dereferenceable(1) %5, ptr noalias align 256 dereferenceable(992) %6, ptr noalias align 256 dereferenceable(512) %7, ptr noalias align 256 dereferenceable(1536) %8, ptr noalias align 256 dereferenceable(65536) %9, ptr noalias align 256 dereferenceable(65536) %10) #0 {
  %12 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !1
  %13 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %14 = mul i32 %12, 2
  %15 = udiv i32 %13, 64
  %16 = add i32 %14, %15
  %17 = urem i32 %13, 64
  %18 = call { double, double } @fused_add_1_add_333_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7, ptr %8, i32 %16, i32 %17)
  %19 = icmp slt i32 %16, 62
  br i1 %19, label %20, label %44

20:                                               ; preds = %11
  %21 = add i32 %16, 2
  %22 = call { double, double } @fused_add_1_complex_29_20_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7, ptr %8, i32 %21, i32 %17)
  %23 = getelementptr inbounds [62 x { double, double }], ptr %4, i32 0, i32 %16
  %24 = load { double, double }, ptr %23, align 8, !invariant.load !3
  %25 = extractvalue { double, double } %22, 0
  %26 = extractvalue { double, double } %22, 1
  %27 = extractvalue { double, double } %24, 0
  %28 = extractvalue { double, double } %24, 1
  %29 = fmul double %25, %27
  %30 = fmul double %26, %28
  %31 = fmul double %26, %27
  %32 = fmul double %25, %28
  %33 = fsub double %29, %30
  %34 = fadd double %31, %32
  %35 = getelementptr inbounds [1 x i8], ptr %5, i32 0, i32 0
  %36 = load i8, ptr %35, align 1, !invariant.load !3
  %37 = call double @fused_add_1_real_46_5_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7, ptr %8, i32 %16, i32 %17)
  %38 = fadd double %37, %33
  %39 = trunc i8 %36 to i1
  %40 = select i1 %39, double %38, double %37
  %41 = call double @fused_add_1_imag_37_5_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7, ptr %8, i32 %16, i32 %17)
  %42 = fadd double %41, %34
  %43 = select i1 %39, double %42, double %41
  br label %47

44:                                               ; preds = %11
  %45 = call double @fused_add_1_real_46_5_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7, ptr %8, i32 %16, i32 %17)
  %46 = call double @fused_add_1_imag_37_5_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7, ptr %8, i32 %16, i32 %17)
  br label %47

47:                                               ; preds = %20, %44
  %48 = phi double [ %45, %44 ], [ %40, %20 ]
  %49 = phi double [ %46, %44 ], [ %43, %20 ]
  br label %50

50:                                               ; preds = %47
  %51 = icmp sle i32 %16, 62
  br i1 %51, label %52, label %85

52:                                               ; preds = %50
  %53 = add i32 %16, 1
  %54 = icmp sle i32 %17, 62
  br i1 %54, label %55, label %70

55:                                               ; preds = %52
  %56 = getelementptr inbounds [63 x { double, double }], ptr %3, i32 0, i32 %17
  %57 = load { double, double }, ptr %56, align 8, !invariant.load !3
  %58 = add i32 %17, 1
  %59 = call { double, double } @fused_add_1_add_333_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7, ptr %8, i32 %53, i32 %58)
  %60 = extractvalue { double, double } %57, 0
  %61 = extractvalue { double, double } %57, 1
  %62 = extractvalue { double, double } %59, 0
  %63 = extractvalue { double, double } %59, 1
  %64 = fmul double %60, %62
  %65 = fmul double %61, %63
  %66 = fmul double %61, %62
  %67 = fmul double %60, %63
  %68 = fsub double %64, %65
  %69 = fadd double %66, %67
  br label %71

70:                                               ; preds = %52
  br label %71

71:                                               ; preds = %55, %70
  %72 = phi double [ 0.000000e+00, %70 ], [ %68, %55 ]
  %73 = phi double [ 0.000000e+00, %70 ], [ %69, %55 ]
  br label %74

74:                                               ; preds = %71
  %75 = getelementptr inbounds [63 x { double, double }], ptr %2, i32 0, i32 %16
  %76 = load { double, double }, ptr %75, align 8, !invariant.load !3
  %77 = extractvalue { double, double } %76, 0
  %78 = extractvalue { double, double } %76, 1
  %79 = fmul double %72, %77
  %80 = fmul double %73, %78
  %81 = fmul double %73, %77
  %82 = fmul double %72, %78
  %83 = fsub double %79, %80
  %84 = fadd double %81, %82
  br label %86

85:                                               ; preds = %50
  br label %86

86:                                               ; preds = %74, %85
  %87 = phi double [ 0.000000e+00, %85 ], [ %83, %74 ]
  %88 = phi double [ 0.000000e+00, %85 ], [ %84, %74 ]
  br label %89

89:                                               ; preds = %86
  %90 = icmp slt i32 %17, 62
  br i1 %90, label %91, label %109

91:                                               ; preds = %89
  %92 = add i32 %17, 2
  %93 = call { double, double } @fused_add_1_scatter_add_627_7_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7, ptr %8, i32 2, i32 %92)
  %94 = call { double, double } @fused_add_1_add_333_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7, ptr %8, i32 %16, i32 %92)
  %95 = extractvalue { double, double } %93, 0
  %96 = extractvalue { double, double } %93, 1
  %97 = extractvalue { double, double } %94, 0
  %98 = extractvalue { double, double } %94, 1
  %99 = fmul double %95, %97
  %100 = fmul double %96, %98
  %101 = fmul double %96, %97
  %102 = fmul double %95, %98
  %103 = fsub double %99, %100
  %104 = fadd double %101, %102
  %105 = call double @fused_add_1_add_334_3_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7, ptr %8, i32 %16, i32 %17)
  %106 = fadd double %105, %103
  %107 = call double @fused_add_1_add_336_3_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7, ptr %8, i32 %16, i32 %17)
  %108 = fadd double %107, %104
  br label %112

109:                                              ; preds = %89
  %110 = call double @fused_add_1_add_334_3_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7, ptr %8, i32 %16, i32 %17)
  %111 = call double @fused_add_1_add_336_3_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7, ptr %8, i32 %16, i32 %17)
  br label %112

112:                                              ; preds = %91, %109
  %113 = phi double [ %110, %109 ], [ %106, %91 ]
  %114 = phi double [ %111, %109 ], [ %108, %91 ]
  br label %115

115:                                              ; preds = %112
  %116 = fadd double %48, %87
  %117 = fadd double %49, %88
  %118 = fmul double %116, 5.000000e-01
  %119 = fmul double %117, 0.000000e+00
  %120 = fmul double %117, 5.000000e-01
  %121 = fmul double %116, 0.000000e+00
  %122 = fsub double %118, %119
  %123 = fadd double %120, %121
  %124 = fadd double %113, %122
  %125 = fadd double %114, %123
  %126 = insertvalue { double, double } poison, double %124, 0
  %127 = insertvalue { double, double } %126, double %125, 1
  %128 = mul i32 %12, 128
  %129 = add i32 %128, %13
  %130 = getelementptr inbounds [4096 x { double, double }], ptr %9, i32 0, i32 %129
  store { double, double } %18, ptr %130, align 8
  %131 = getelementptr inbounds [4096 x { double, double }], ptr %10, i32 0, i32 %129
  store { double, double } %127, ptr %131, align 8
  ret void
}

define internal double @fused_add_1_imag_37_5_clone_1(ptr noalias %0, ptr noalias %1, ptr noalias %2, ptr noalias %3, ptr noalias %4, ptr noalias %5, ptr noalias %6, ptr noalias %7, ptr noalias %8, i32 %9, i32 %10) {
  %12 = call { double, double } @fused_add_1_multiply_72_13_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7, ptr %8, i32 %9, i32 %10)
  %13 = extractvalue { double, double } %12, 1
  ret double %13
}

define internal double @fused_add_1_real_46_5_clone_1(ptr noalias %0, ptr noalias %1, ptr noalias %2, ptr noalias %3, ptr noalias %4, ptr noalias %5, ptr noalias %6, ptr noalias %7, ptr noalias %8, i32 %9, i32 %10) {
  %12 = call { double, double } @fused_add_1_multiply_72_13_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7, ptr %8, i32 %9, i32 %10)
  %13 = extractvalue { double, double } %12, 0
  ret double %13
}

define internal { double, double } @fused_add_1_multiply_72_13_clone_1(ptr noalias %0, ptr noalias %1, ptr noalias %2, ptr noalias %3, ptr noalias %4, ptr noalias %5, ptr noalias %6, ptr noalias %7, ptr noalias %8, i32 %9, i32 %10) {
  %12 = call { double, double } @fused_add_1_complex_29_20_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7, ptr %8, i32 %9, i32 %10)
  %13 = extractvalue { double, double } %12, 0
  %14 = extractvalue { double, double } %12, 1
  %15 = fmul double %13, -4.000000e+00
  %16 = fmul double %14, -0.000000e+00
  %17 = fmul double %14, -4.000000e+00
  %18 = fmul double %13, -0.000000e+00
  %19 = fsub double %15, %16
  %20 = fadd double %17, %18
  %21 = insertvalue { double, double } poison, double %19, 0
  %22 = insertvalue { double, double } %21, double %20, 1
  ret { double, double } %22
}

define internal { double, double } @fused_add_1_complex_29_20_clone_1(ptr noalias %0, ptr noalias %1, ptr noalias %2, ptr noalias %3, ptr noalias %4, ptr noalias %5, ptr noalias %6, ptr noalias %7, ptr noalias %8, i32 %9, i32 %10) {
  %12 = icmp slt i32 %10, 62
  br i1 %12, label %13, label %32

13:                                               ; preds = %11
  %14 = getelementptr inbounds [62 x { double, double }], ptr %6, i32 0, i32 %10
  %15 = load { double, double }, ptr %14, align 8, !invariant.load !3
  %16 = add i32 %10, 2
  %17 = call { double, double } @fused_add_1_add_333_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7, ptr %8, i32 %9, i32 %16)
  %18 = extractvalue { double, double } %15, 0
  %19 = extractvalue { double, double } %15, 1
  %20 = extractvalue { double, double } %17, 0
  %21 = extractvalue { double, double } %17, 1
  %22 = fmul double %18, %20
  %23 = fmul double %19, %21
  %24 = fmul double %19, %20
  %25 = fmul double %18, %21
  %26 = fsub double %22, %23
  %27 = fadd double %24, %25
  %28 = call double @fused_add_1_real_44_5_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7, ptr %8, i32 %9, i32 %10)
  %29 = fadd double %28, %26
  %30 = call double @fused_add_1_imag_35_5_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7, ptr %8, i32 %9, i32 %10)
  %31 = fadd double %30, %27
  br label %35

32:                                               ; preds = %11
  %33 = call double @fused_add_1_real_44_5_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7, ptr %8, i32 %9, i32 %10)
  %34 = call double @fused_add_1_imag_35_5_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7, ptr %8, i32 %9, i32 %10)
  br label %35

35:                                               ; preds = %13, %32
  %36 = phi double [ %33, %32 ], [ %29, %13 ]
  %37 = phi double [ %34, %32 ], [ %31, %13 ]
  br label %38

38:                                               ; preds = %35
  %39 = insertvalue { double, double } poison, double %36, 0
  %40 = insertvalue { double, double } %39, double %37, 1
  ret { double, double } %40
}

define internal double @fused_add_1_imag_35_5_clone_1(ptr noalias %0, ptr noalias %1, ptr noalias %2, ptr noalias %3, ptr noalias %4, ptr noalias %5, ptr noalias %6, ptr noalias %7, ptr noalias %8, i32 %9, i32 %10) {
  %12 = call { double, double } @fused_add_1_multiply_70_9_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7, ptr %8, i32 %9, i32 %10)
  %13 = extractvalue { double, double } %12, 1
  ret double %13
}

define internal double @fused_add_1_real_44_5_clone_1(ptr noalias %0, ptr noalias %1, ptr noalias %2, ptr noalias %3, ptr noalias %4, ptr noalias %5, ptr noalias %6, ptr noalias %7, ptr noalias %8, i32 %9, i32 %10) {
  %12 = call { double, double } @fused_add_1_multiply_70_9_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7, ptr %8, i32 %9, i32 %10)
  %13 = extractvalue { double, double } %12, 0
  ret double %13
}

define internal { double, double } @fused_add_1_multiply_70_9_clone_1(ptr noalias %0, ptr noalias %1, ptr noalias %2, ptr noalias %3, ptr noalias %4, ptr noalias %5, ptr noalias %6, ptr noalias %7, ptr noalias %8, i32 %9, i32 %10) {
  %12 = call { double, double } @fused_add_1_add_333_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7, ptr %8, i32 %9, i32 %10)
  %13 = extractvalue { double, double } %12, 0
  %14 = extractvalue { double, double } %12, 1
  %15 = fmul double %13, -4.000000e+00
  %16 = fmul double %14, 0.000000e+00
  %17 = fmul double %13, 0.000000e+00
  %18 = fmul double %14, -4.000000e+00
  %19 = fsub double %15, %16
  %20 = fadd double %17, %18
  %21 = insertvalue { double, double } poison, double %19, 0
  %22 = insertvalue { double, double } %21, double %20, 1
  ret { double, double } %22
}

define internal double @fused_add_1_add_336_3_clone_1(ptr noalias %0, ptr noalias %1, ptr noalias %2, ptr noalias %3, ptr noalias %4, ptr noalias %5, ptr noalias %6, ptr noalias %7, ptr noalias %8, i32 %9, i32 %10) {
  %12 = icmp sge i32 %10, 2
  br i1 %12, label %13, label %17

13:                                               ; preds = %11
  %14 = add i32 %10, -2
  %15 = call { double, double } @fused_add_1_multiply_67_13_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7, ptr %8, i32 %9, i32 %14)
  %16 = extractvalue { double, double } %15, 1
  br label %18

17:                                               ; preds = %11
  br label %18

18:                                               ; preds = %13, %17
  %19 = phi double [ 0.000000e+00, %17 ], [ %16, %13 ]
  br label %20

20:                                               ; preds = %18
  %21 = call { double, double } @fused_add_1_multiply_68_13_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7, ptr %8, i32 %9, i32 %10)
  %22 = extractvalue { double, double } %21, 1
  %23 = fadd double %19, %22
  ret double %23
}

define internal double @fused_add_1_add_334_3_clone_1(ptr noalias %0, ptr noalias %1, ptr noalias %2, ptr noalias %3, ptr noalias %4, ptr noalias %5, ptr noalias %6, ptr noalias %7, ptr noalias %8, i32 %9, i32 %10) {
  %12 = icmp sge i32 %10, 2
  br i1 %12, label %13, label %17

13:                                               ; preds = %11
  %14 = add i32 %10, -2
  %15 = call { double, double } @fused_add_1_multiply_67_13_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7, ptr %8, i32 %9, i32 %14)
  %16 = extractvalue { double, double } %15, 0
  br label %18

17:                                               ; preds = %11
  br label %18

18:                                               ; preds = %13, %17
  %19 = phi double [ 0.000000e+00, %17 ], [ %16, %13 ]
  br label %20

20:                                               ; preds = %18
  %21 = call { double, double } @fused_add_1_multiply_68_13_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7, ptr %8, i32 %9, i32 %10)
  %22 = extractvalue { double, double } %21, 0
  %23 = fadd double %19, %22
  ret double %23
}

define internal { double, double } @fused_add_1_multiply_68_13_clone_1(ptr noalias %0, ptr noalias %1, ptr noalias %2, ptr noalias %3, ptr noalias %4, ptr noalias %5, ptr noalias %6, ptr noalias %7, ptr noalias %8, i32 %9, i32 %10) {
  %12 = call { double, double } @fused_add_1_scatter_add_627_7_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7, ptr %8, i32 1, i32 %10)
  %13 = call { double, double } @fused_add_1_add_333_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7, ptr %8, i32 %9, i32 %10)
  %14 = extractvalue { double, double } %12, 0
  %15 = extractvalue { double, double } %12, 1
  %16 = extractvalue { double, double } %13, 0
  %17 = extractvalue { double, double } %13, 1
  %18 = fmul double %14, %16
  %19 = fmul double %15, %17
  %20 = fmul double %15, %16
  %21 = fmul double %14, %17
  %22 = fsub double %18, %19
  %23 = fadd double %20, %21
  %24 = insertvalue { double, double } poison, double %22, 0
  %25 = insertvalue { double, double } %24, double %23, 1
  ret { double, double } %25
}

define internal { double, double } @fused_add_1_multiply_67_13_clone_1(ptr noalias %0, ptr noalias %1, ptr noalias %2, ptr noalias %3, ptr noalias %4, ptr noalias %5, ptr noalias %6, ptr noalias %7, ptr noalias %8, i32 %9, i32 %10) {
  %12 = call { double, double } @fused_add_1_scatter_add_627_7_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7, ptr %8, i32 0, i32 %10)
  %13 = call { double, double } @fused_add_1_add_333_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7, ptr %8, i32 %9, i32 %10)
  %14 = extractvalue { double, double } %12, 0
  %15 = extractvalue { double, double } %12, 1
  %16 = extractvalue { double, double } %13, 0
  %17 = extractvalue { double, double } %13, 1
  %18 = fmul double %14, %16
  %19 = fmul double %15, %17
  %20 = fmul double %15, %16
  %21 = fmul double %14, %17
  %22 = fsub double %18, %19
  %23 = fadd double %20, %21
  %24 = insertvalue { double, double } poison, double %22, 0
  %25 = insertvalue { double, double } %24, double %23, 1
  ret { double, double } %25
}

define internal { double, double } @fused_add_1_scatter_add_627_7_clone_1(ptr noalias %0, ptr noalias %1, ptr noalias %2, ptr noalias %3, ptr noalias %4, ptr noalias %5, ptr noalias %6, ptr noalias %7, ptr noalias %8, i32 %9, i32 %10) {
  %12 = icmp sge i32 %9, 2
  br i1 %12, label %13, label %20

13:                                               ; preds = %11
  %14 = call double @fused_add_1_dynamic_update_slice_198_8_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7, ptr %8, i32 2, i32 %10)
  %15 = call double @fused_add_1_scatter_add_663_3_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7, ptr %8, i32 2, i32 %10)
  %16 = fadd double %14, %15
  %17 = call double @fused_add_1_dynamic_update_slice_201_8_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7, ptr %8, i32 2, i32 %10)
  %18 = call double @fused_add_1_scatter_add_664_3_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7, ptr %8, i32 2, i32 %10)
  %19 = fadd double %17, %18
  br label %23

20:                                               ; preds = %11
  %21 = call double @fused_add_1_dynamic_update_slice_198_8_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7, ptr %8, i32 %9, i32 %10)
  %22 = call double @fused_add_1_dynamic_update_slice_201_8_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7, ptr %8, i32 %9, i32 %10)
  br label %23

23:                                               ; preds = %13, %20
  %24 = phi double [ %21, %20 ], [ %16, %13 ]
  %25 = phi double [ %22, %20 ], [ %19, %13 ]
  br label %26

26:                                               ; preds = %23
  %27 = insertvalue { double, double } poison, double %24, 0
  %28 = insertvalue { double, double } %27, double %25, 1
  ret { double, double } %28
}

define internal double @fused_add_1_dynamic_update_slice_201_8_clone_1(ptr noalias %0, ptr noalias %1, ptr noalias %2, ptr noalias %3, ptr noalias %4, ptr noalias %5, ptr noalias %6, ptr noalias %7, ptr noalias %8, i32 %9, i32 %10) {
  %12 = icmp sge i32 %9, 1
  %13 = icmp slt i32 %9, 2
  %14 = and i1 %12, %13
  br i1 %14, label %15, label %19

15:                                               ; preds = %11
  %16 = call double @fused_add_1_dynamic_update_slice_200_9_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7, ptr %8, i32 1, i32 %10)
  %17 = call double @fused_add_1_scatter_add_664_3_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7, ptr %8, i32 1, i32 %10)
  %18 = fadd double %16, %17
  br label %21

19:                                               ; preds = %11
  %20 = call double @fused_add_1_dynamic_update_slice_200_9_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7, ptr %8, i32 %9, i32 %10)
  br label %21

21:                                               ; preds = %15, %19
  %22 = phi double [ %20, %19 ], [ %18, %15 ]
  br label %23

23:                                               ; preds = %21
  ret double %22
}

define internal double @fused_add_1_dynamic_update_slice_200_9_clone_1(ptr noalias %0, ptr noalias %1, ptr noalias %2, ptr noalias %3, ptr noalias %4, ptr noalias %5, ptr noalias %6, ptr noalias %7, ptr noalias %8, i32 %9, i32 %10) {
  %12 = icmp sge i32 %9, 1
  %13 = icmp slt i32 %9, 2
  %14 = and i1 %12, %13
  br i1 %14, label %15, label %18

15:                                               ; preds = %11
  %16 = getelementptr inbounds [64 x double], ptr %7, i32 0, i32 %10
  %17 = load double, ptr %16, align 8, !invariant.load !3
  br label %26

18:                                               ; preds = %11
  %19 = icmp eq i32 %9, 0
  br i1 %19, label %20, label %22

20:                                               ; preds = %18
  %21 = call double @fused_add_1_scatter_add_664_3_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7, ptr %8, i32 %9, i32 %10)
  br label %23

22:                                               ; preds = %18
  br label %23

23:                                               ; preds = %20, %22
  %24 = phi double [ 0.000000e+00, %22 ], [ %21, %20 ]
  br label %25

25:                                               ; preds = %23
  br label %26

26:                                               ; preds = %15, %25
  %27 = phi double [ %24, %25 ], [ %17, %15 ]
  br label %28

28:                                               ; preds = %26
  ret double %27
}

define internal double @fused_add_1_scatter_add_664_3_clone_1(ptr noalias %0, ptr noalias %1, ptr noalias %2, ptr noalias %3, ptr noalias %4, ptr noalias %5, ptr noalias %6, ptr noalias %7, ptr noalias %8, i32 %9, i32 %10) {
  %12 = call { double, double } @fused_add_1_mul_333_3_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7, ptr %8, i32 %9, i32 %10)
  %13 = extractvalue { double, double } %12, 1
  ret double %13
}

define internal double @fused_add_1_dynamic_update_slice_198_8_clone_1(ptr noalias %0, ptr noalias %1, ptr noalias %2, ptr noalias %3, ptr noalias %4, ptr noalias %5, ptr noalias %6, ptr noalias %7, ptr noalias %8, i32 %9, i32 %10) {
  %12 = icmp sge i32 %9, 1
  %13 = icmp slt i32 %9, 2
  %14 = and i1 %12, %13
  br i1 %14, label %15, label %19

15:                                               ; preds = %11
  %16 = call double @fused_add_1_dynamic_update_slice_197_9_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7, ptr %8, i32 1, i32 %10)
  %17 = call double @fused_add_1_scatter_add_663_3_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7, ptr %8, i32 1, i32 %10)
  %18 = fadd double %16, %17
  br label %21

19:                                               ; preds = %11
  %20 = call double @fused_add_1_dynamic_update_slice_197_9_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7, ptr %8, i32 %9, i32 %10)
  br label %21

21:                                               ; preds = %15, %19
  %22 = phi double [ %20, %19 ], [ %18, %15 ]
  br label %23

23:                                               ; preds = %21
  ret double %22
}

define internal double @fused_add_1_dynamic_update_slice_197_9_clone_1(ptr noalias %0, ptr noalias %1, ptr noalias %2, ptr noalias %3, ptr noalias %4, ptr noalias %5, ptr noalias %6, ptr noalias %7, ptr noalias %8, i32 %9, i32 %10) {
  %12 = icmp sge i32 %9, 1
  %13 = icmp slt i32 %9, 2
  %14 = and i1 %12, %13
  br i1 %14, label %15, label %16

15:                                               ; preds = %11
  br label %24

16:                                               ; preds = %11
  %17 = icmp eq i32 %9, 0
  br i1 %17, label %18, label %20

18:                                               ; preds = %16
  %19 = call double @fused_add_1_scatter_add_663_3_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7, ptr %8, i32 %9, i32 %10)
  br label %21

20:                                               ; preds = %16
  br label %21

21:                                               ; preds = %18, %20
  %22 = phi double [ 0.000000e+00, %20 ], [ %19, %18 ]
  br label %23

23:                                               ; preds = %21
  br label %24

24:                                               ; preds = %15, %23
  %25 = phi double [ %22, %23 ], [ 0.000000e+00, %15 ]
  br label %26

26:                                               ; preds = %24
  ret double %25
}

define internal double @fused_add_1_scatter_add_663_3_clone_1(ptr noalias %0, ptr noalias %1, ptr noalias %2, ptr noalias %3, ptr noalias %4, ptr noalias %5, ptr noalias %6, ptr noalias %7, ptr noalias %8, i32 %9, i32 %10) {
  %12 = call { double, double } @fused_add_1_mul_333_3_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7, ptr %8, i32 %9, i32 %10)
  %13 = extractvalue { double, double } %12, 0
  ret double %13
}

define internal { double, double } @fused_add_1_mul_333_3_clone_1(ptr noalias %0, ptr noalias %1, ptr noalias %2, ptr noalias %3, ptr noalias %4, ptr noalias %5, ptr noalias %6, ptr noalias %7, ptr noalias %8, i32 %9, i32 %10) {
  %12 = mul i32 %9, 64
  %13 = add i32 %12, %10
  %14 = getelementptr inbounds [192 x double], ptr %8, i32 0, i32 %13
  %15 = load double, ptr %14, align 8, !invariant.load !3
  %16 = icmp eq i32 %9, 2
  br i1 %16, label %17, label %20

17:                                               ; preds = %11
  %18 = icmp sge i32 %10, 2
  %19 = select i1 %18, double -0.000000e+00, double 0.000000e+00
  br label %21

20:                                               ; preds = %11
  br label %21

21:                                               ; preds = %17, %20
  %22 = phi double [ 0.000000e+00, %20 ], [ %19, %17 ]
  br label %23

23:                                               ; preds = %21
  %24 = fmul double %15, -5.000000e-01
  %25 = fmul double %22, 0.000000e+00
  %26 = fmul double %22, -5.000000e-01
  %27 = fmul double %15, 0.000000e+00
  %28 = fsub double %24, %25
  %29 = fadd double %26, %27
  %30 = insertvalue { double, double } poison, double %28, 0
  %31 = insertvalue { double, double } %30, double %29, 1
  ret { double, double } %31
}

define internal { double, double } @fused_add_1_add_333_1(ptr noalias %0, ptr noalias %1, ptr noalias %2, ptr noalias %3, ptr noalias %4, ptr noalias %5, ptr noalias %6, ptr noalias %7, ptr noalias %8, i32 %9, i32 %10) {
  %12 = mul i32 %9, 64
  %13 = add i32 %12, %10
  %14 = getelementptr inbounds [4096 x { double, double }], ptr %0, i32 0, i32 %13
  %15 = load { double, double }, ptr %14, align 8, !invariant.load !3
  %16 = mul i32 %10, 64
  %17 = add i32 %16, %9
  %18 = getelementptr inbounds [4096 x { double, double }], ptr %1, i32 0, i32 %17
  %19 = load { double, double }, ptr %18, align 8, !invariant.load !3
  %20 = extractvalue { double, double } %15, 0
  %21 = extractvalue { double, double } %19, 0
  %22 = fadd double %20, %21
  %23 = extractvalue { double, double } %15, 1
  %24 = extractvalue { double, double } %19, 1
  %25 = fadd double %23, %24
  %26 = insertvalue { double, double } poison, double %22, 0
  %27 = insertvalue { double, double } %26, double %25, 1
  ret { double, double } %27
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

attributes #0 = { "nvvm.reqntid"="128,1,1" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 0, i32 32}
!2 = !{i32 0, i32 128}
!3 = !{}
