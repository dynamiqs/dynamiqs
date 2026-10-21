; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @input_add_fusion(ptr noalias align 256 dereferenceable(65536) %0, ptr noalias align 256 dereferenceable(65536) %1, ptr noalias align 16 dereferenceable(1024) %2, ptr noalias align 256 dereferenceable(1) %3, ptr noalias align 16 dereferenceable(2048) %4, ptr noalias align 256 dereferenceable(3072) %5, ptr noalias align 16 dereferenceable(1024) %6, ptr noalias align 256 dereferenceable(65536) %7, ptr noalias align 256 dereferenceable(65536) %8) #0 {
  %10 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !1
  %11 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %12 = mul i32 %10, 2
  %13 = udiv i32 %11, 64
  %14 = add i32 %12, %13
  %15 = urem i32 %11, 64
  %16 = call { double, double } @fused_add_add_416_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, i32 %14, i32 %15)
  %17 = icmp slt i32 %14, 62
  br i1 %17, label %18, label %41

18:                                               ; preds = %9
  %19 = add i32 %14, 2
  %20 = call { double, double } @fused_add_complex_36_16_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, i32 %19, i32 %15)
  %21 = call { double, double } @fused_add_complex_32_16_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, i32 1, i32 %14)
  %22 = extractvalue { double, double } %20, 0
  %23 = extractvalue { double, double } %20, 1
  %24 = extractvalue { double, double } %21, 0
  %25 = extractvalue { double, double } %21, 1
  %26 = fmul double %22, %24
  %27 = fmul double %23, %25
  %28 = fmul double %23, %24
  %29 = fmul double %22, %25
  %30 = fsub double %26, %27
  %31 = fadd double %28, %29
  %32 = getelementptr inbounds [1 x i8], ptr %3, i32 0, i32 0
  %33 = load i8, ptr %32, align 1, !invariant.load !3
  %34 = call double @fused_add_real_53_5_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, i32 %14, i32 %15)
  %35 = fadd double %34, %30
  %36 = trunc i8 %33 to i1
  %37 = select i1 %36, double %35, double %34
  %38 = call double @fused_add_imag_43_5_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, i32 %14, i32 %15)
  %39 = fadd double %38, %31
  %40 = select i1 %36, double %39, double %38
  br label %44

41:                                               ; preds = %9
  %42 = call double @fused_add_real_53_5_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, i32 %14, i32 %15)
  %43 = call double @fused_add_imag_43_5_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, i32 %14, i32 %15)
  br label %44

44:                                               ; preds = %18, %41
  %45 = phi double [ %42, %41 ], [ %37, %18 ]
  %46 = phi double [ %43, %41 ], [ %40, %18 ]
  br label %47

47:                                               ; preds = %44
  %48 = icmp sle i32 %14, 62
  br i1 %48, label %49, label %83

49:                                               ; preds = %47
  %50 = add i32 %14, 1
  %51 = icmp sle i32 %15, 62
  br i1 %51, label %52, label %67

52:                                               ; preds = %49
  %53 = add i32 %15, 1
  %54 = getelementptr inbounds [64 x { double, double }], ptr %2, i32 0, i32 %53
  %55 = load { double, double }, ptr %54, align 8, !invariant.load !3
  %56 = call { double, double } @fused_add_add_416_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, i32 %50, i32 %53)
  %57 = extractvalue { double, double } %55, 0
  %58 = extractvalue { double, double } %55, 1
  %59 = extractvalue { double, double } %56, 0
  %60 = extractvalue { double, double } %56, 1
  %61 = fmul double %57, %59
  %62 = fmul double %58, %60
  %63 = fmul double %58, %59
  %64 = fmul double %57, %60
  %65 = fsub double %61, %62
  %66 = fadd double %63, %64
  br label %68

67:                                               ; preds = %49
  br label %68

68:                                               ; preds = %52, %67
  %69 = phi double [ 0.000000e+00, %67 ], [ %65, %52 ]
  %70 = phi double [ 0.000000e+00, %67 ], [ %66, %52 ]
  br label %71

71:                                               ; preds = %68
  %72 = getelementptr inbounds [64 x { double, double }], ptr %2, i32 0, i32 %50
  %73 = load { double, double }, ptr %72, align 8, !invariant.load !3
  %74 = extractvalue { double, double } %73, 1
  %75 = extractvalue { double, double } %73, 0
  %76 = fneg double %74
  %77 = fmul double %69, %75
  %78 = fmul double %70, %76
  %79 = fmul double %70, %75
  %80 = fmul double %69, %76
  %81 = fsub double %77, %78
  %82 = fadd double %79, %80
  br label %84

83:                                               ; preds = %47
  br label %84

84:                                               ; preds = %71, %83
  %85 = phi double [ 0.000000e+00, %83 ], [ %81, %71 ]
  %86 = phi double [ 0.000000e+00, %83 ], [ %82, %71 ]
  br label %87

87:                                               ; preds = %84
  %88 = icmp slt i32 %15, 62
  br i1 %88, label %89, label %107

89:                                               ; preds = %87
  %90 = add i32 %15, 2
  %91 = call { double, double } @fused_add_scatter_add_646_7_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, i32 2, i32 %90)
  %92 = call { double, double } @fused_add_add_416_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, i32 %14, i32 %90)
  %93 = extractvalue { double, double } %91, 0
  %94 = extractvalue { double, double } %91, 1
  %95 = extractvalue { double, double } %92, 0
  %96 = extractvalue { double, double } %92, 1
  %97 = fmul double %93, %95
  %98 = fmul double %94, %96
  %99 = fmul double %94, %95
  %100 = fmul double %93, %96
  %101 = fsub double %97, %98
  %102 = fadd double %99, %100
  %103 = call double @fused_add_add_417_3_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, i32 %14, i32 %15)
  %104 = fadd double %103, %101
  %105 = call double @fused_add_add_419_3_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, i32 %14, i32 %15)
  %106 = fadd double %105, %102
  br label %110

107:                                              ; preds = %87
  %108 = call double @fused_add_add_417_3_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, i32 %14, i32 %15)
  %109 = call double @fused_add_add_419_3_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, i32 %14, i32 %15)
  br label %110

110:                                              ; preds = %89, %107
  %111 = phi double [ %108, %107 ], [ %104, %89 ]
  %112 = phi double [ %109, %107 ], [ %106, %89 ]
  br label %113

113:                                              ; preds = %110
  %114 = fadd double %45, %85
  %115 = fadd double %46, %86
  %116 = fmul double %114, 5.000000e-01
  %117 = fmul double %115, 0.000000e+00
  %118 = fmul double %115, 5.000000e-01
  %119 = fmul double %114, 0.000000e+00
  %120 = fsub double %116, %117
  %121 = fadd double %118, %119
  %122 = fadd double %111, %120
  %123 = fadd double %112, %121
  %124 = insertvalue { double, double } poison, double %122, 0
  %125 = insertvalue { double, double } %124, double %123, 1
  %126 = mul i32 %10, 128
  %127 = add i32 %126, %11
  %128 = getelementptr inbounds [4096 x { double, double }], ptr %7, i32 0, i32 %127
  store { double, double } %16, ptr %128, align 8
  %129 = getelementptr inbounds [4096 x { double, double }], ptr %8, i32 0, i32 %127
  store { double, double } %125, ptr %129, align 8
  ret void
}

define internal double @fused_add_imag_43_5_clone_1(ptr noalias %0, ptr noalias %1, ptr noalias %2, ptr noalias %3, ptr noalias %4, ptr noalias %5, ptr noalias %6, i32 %7, i32 %8) {
  %10 = call { double, double } @fused_add_multiply_84_9_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, i32 %7, i32 %8)
  %11 = extractvalue { double, double } %10, 1
  ret double %11
}

define internal double @fused_add_real_53_5_clone_1(ptr noalias %0, ptr noalias %1, ptr noalias %2, ptr noalias %3, ptr noalias %4, ptr noalias %5, ptr noalias %6, i32 %7, i32 %8) {
  %10 = call { double, double } @fused_add_multiply_84_9_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, i32 %7, i32 %8)
  %11 = extractvalue { double, double } %10, 0
  ret double %11
}

define internal { double, double } @fused_add_multiply_84_9_clone_1(ptr noalias %0, ptr noalias %1, ptr noalias %2, ptr noalias %3, ptr noalias %4, ptr noalias %5, ptr noalias %6, i32 %7, i32 %8) {
  %10 = call { double, double } @fused_add_complex_32_16_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, i32 0, i32 %7)
  %11 = call { double, double } @fused_add_complex_36_16_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, i32 %7, i32 %8)
  %12 = extractvalue { double, double } %11, 0
  %13 = extractvalue { double, double } %11, 1
  %14 = extractvalue { double, double } %10, 0
  %15 = extractvalue { double, double } %10, 1
  %16 = fmul double %12, %14
  %17 = fmul double %13, %15
  %18 = fmul double %13, %14
  %19 = fmul double %12, %15
  %20 = fsub double %16, %17
  %21 = fadd double %18, %19
  %22 = insertvalue { double, double } poison, double %20, 0
  %23 = insertvalue { double, double } %22, double %21, 1
  ret { double, double } %23
}

define internal { double, double } @fused_add_complex_32_16_clone_1(ptr noalias %0, ptr noalias %1, ptr noalias %2, ptr noalias %3, ptr noalias %4, ptr noalias %5, ptr noalias %6, i32 %7, i32 %8) {
  %10 = icmp sge i32 %7, 1
  %11 = icmp slt i32 %8, 62
  %12 = and i1 %10, %11
  br i1 %12, label %13, label %17

13:                                               ; preds = %9
  %14 = add i32 %8, 66
  %15 = getelementptr inbounds [128 x { double, double }], ptr %4, i32 0, i32 %14
  %16 = load { double, double }, ptr %15, align 8, !invariant.load !3
  br label %28

17:                                               ; preds = %9
  %18 = icmp eq i32 %7, 0
  br i1 %18, label %19, label %24

19:                                               ; preds = %17
  %20 = mul i32 %7, 64
  %21 = add i32 %20, %8
  %22 = getelementptr inbounds [128 x { double, double }], ptr %4, i32 0, i32 %21
  %23 = load { double, double }, ptr %22, align 8, !invariant.load !3
  br label %25

24:                                               ; preds = %17
  br label %25

25:                                               ; preds = %19, %24
  %26 = phi { double, double } [ zeroinitializer, %24 ], [ %23, %19 ]
  br label %27

27:                                               ; preds = %25
  br label %28

28:                                               ; preds = %13, %27
  %29 = phi { double, double } [ %26, %27 ], [ %16, %13 ]
  br label %30

30:                                               ; preds = %28
  %31 = extractvalue { double, double } %29, 1
  %32 = extractvalue { double, double } %29, 0
  %33 = fneg double %31
  %34 = insertvalue { double, double } poison, double %32, 0
  %35 = insertvalue { double, double } %34, double %33, 1
  ret { double, double } %35
}

define internal { double, double } @fused_add_complex_36_16_clone_1(ptr noalias %0, ptr noalias %1, ptr noalias %2, ptr noalias %3, ptr noalias %4, ptr noalias %5, ptr noalias %6, i32 %7, i32 %8) {
  %10 = icmp slt i32 %8, 62
  br i1 %10, label %11, label %31

11:                                               ; preds = %9
  %12 = add i32 %8, 2
  %13 = add i32 %8, 66
  %14 = getelementptr inbounds [128 x { double, double }], ptr %4, i32 0, i32 %13
  %15 = load { double, double }, ptr %14, align 8, !invariant.load !3
  %16 = call { double, double } @fused_add_add_416_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, i32 %7, i32 %12)
  %17 = extractvalue { double, double } %15, 0
  %18 = extractvalue { double, double } %15, 1
  %19 = extractvalue { double, double } %16, 0
  %20 = extractvalue { double, double } %16, 1
  %21 = fmul double %17, %19
  %22 = fmul double %18, %20
  %23 = fmul double %18, %19
  %24 = fmul double %17, %20
  %25 = fsub double %21, %22
  %26 = fadd double %23, %24
  %27 = call double @fused_add_real_51_5_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, i32 %7, i32 %8)
  %28 = fadd double %27, %25
  %29 = call double @fused_add_imag_41_5_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, i32 %7, i32 %8)
  %30 = fadd double %29, %26
  br label %34

31:                                               ; preds = %9
  %32 = call double @fused_add_real_51_5_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, i32 %7, i32 %8)
  %33 = call double @fused_add_imag_41_5_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, i32 %7, i32 %8)
  br label %34

34:                                               ; preds = %11, %31
  %35 = phi double [ %32, %31 ], [ %28, %11 ]
  %36 = phi double [ %33, %31 ], [ %30, %11 ]
  br label %37

37:                                               ; preds = %34
  %38 = insertvalue { double, double } poison, double %35, 0
  %39 = insertvalue { double, double } %38, double %36, 1
  ret { double, double } %39
}

define internal double @fused_add_imag_41_5_clone_1(ptr noalias %0, ptr noalias %1, ptr noalias %2, ptr noalias %3, ptr noalias %4, ptr noalias %5, ptr noalias %6, i32 %7, i32 %8) {
  %10 = call { double, double } @fused_add_multiply_82_9_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, i32 %7, i32 %8)
  %11 = extractvalue { double, double } %10, 1
  ret double %11
}

define internal double @fused_add_real_51_5_clone_1(ptr noalias %0, ptr noalias %1, ptr noalias %2, ptr noalias %3, ptr noalias %4, ptr noalias %5, ptr noalias %6, i32 %7, i32 %8) {
  %10 = call { double, double } @fused_add_multiply_82_9_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, i32 %7, i32 %8)
  %11 = extractvalue { double, double } %10, 0
  ret double %11
}

define internal { double, double } @fused_add_multiply_82_9_clone_1(ptr noalias %0, ptr noalias %1, ptr noalias %2, ptr noalias %3, ptr noalias %4, ptr noalias %5, ptr noalias %6, i32 %7, i32 %8) {
  %10 = getelementptr inbounds [128 x { double, double }], ptr %4, i32 0, i32 %8
  %11 = load { double, double }, ptr %10, align 8, !invariant.load !3
  %12 = call { double, double } @fused_add_add_416_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, i32 %7, i32 %8)
  %13 = extractvalue { double, double } %11, 0
  %14 = extractvalue { double, double } %11, 1
  %15 = extractvalue { double, double } %12, 0
  %16 = extractvalue { double, double } %12, 1
  %17 = fmul double %13, %15
  %18 = fmul double %14, %16
  %19 = fmul double %14, %15
  %20 = fmul double %13, %16
  %21 = fsub double %17, %18
  %22 = fadd double %19, %20
  %23 = insertvalue { double, double } poison, double %21, 0
  %24 = insertvalue { double, double } %23, double %22, 1
  ret { double, double } %24
}

define internal double @fused_add_add_419_3_clone_1(ptr noalias %0, ptr noalias %1, ptr noalias %2, ptr noalias %3, ptr noalias %4, ptr noalias %5, ptr noalias %6, i32 %7, i32 %8) {
  %10 = icmp sge i32 %8, 2
  br i1 %10, label %11, label %15

11:                                               ; preds = %9
  %12 = add i32 %8, -2
  %13 = call { double, double } @fused_add_multiply_79_13_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, i32 %7, i32 %12)
  %14 = extractvalue { double, double } %13, 1
  br label %16

15:                                               ; preds = %9
  br label %16

16:                                               ; preds = %11, %15
  %17 = phi double [ 0.000000e+00, %15 ], [ %14, %11 ]
  br label %18

18:                                               ; preds = %16
  %19 = call { double, double } @fused_add_multiply_80_13_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, i32 %7, i32 %8)
  %20 = extractvalue { double, double } %19, 1
  %21 = fadd double %17, %20
  ret double %21
}

define internal double @fused_add_add_417_3_clone_1(ptr noalias %0, ptr noalias %1, ptr noalias %2, ptr noalias %3, ptr noalias %4, ptr noalias %5, ptr noalias %6, i32 %7, i32 %8) {
  %10 = icmp sge i32 %8, 2
  br i1 %10, label %11, label %15

11:                                               ; preds = %9
  %12 = add i32 %8, -2
  %13 = call { double, double } @fused_add_multiply_79_13_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, i32 %7, i32 %12)
  %14 = extractvalue { double, double } %13, 0
  br label %16

15:                                               ; preds = %9
  br label %16

16:                                               ; preds = %11, %15
  %17 = phi double [ 0.000000e+00, %15 ], [ %14, %11 ]
  br label %18

18:                                               ; preds = %16
  %19 = call { double, double } @fused_add_multiply_80_13_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, i32 %7, i32 %8)
  %20 = extractvalue { double, double } %19, 0
  %21 = fadd double %17, %20
  ret double %21
}

define internal { double, double } @fused_add_multiply_80_13_clone_1(ptr noalias %0, ptr noalias %1, ptr noalias %2, ptr noalias %3, ptr noalias %4, ptr noalias %5, ptr noalias %6, i32 %7, i32 %8) {
  %10 = call { double, double } @fused_add_scatter_add_646_7_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, i32 1, i32 %8)
  %11 = call { double, double } @fused_add_add_416_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, i32 %7, i32 %8)
  %12 = extractvalue { double, double } %10, 0
  %13 = extractvalue { double, double } %10, 1
  %14 = extractvalue { double, double } %11, 0
  %15 = extractvalue { double, double } %11, 1
  %16 = fmul double %12, %14
  %17 = fmul double %13, %15
  %18 = fmul double %13, %14
  %19 = fmul double %12, %15
  %20 = fsub double %16, %17
  %21 = fadd double %18, %19
  %22 = insertvalue { double, double } poison, double %20, 0
  %23 = insertvalue { double, double } %22, double %21, 1
  ret { double, double } %23
}

define internal { double, double } @fused_add_multiply_79_13_clone_1(ptr noalias %0, ptr noalias %1, ptr noalias %2, ptr noalias %3, ptr noalias %4, ptr noalias %5, ptr noalias %6, i32 %7, i32 %8) {
  %10 = call { double, double } @fused_add_scatter_add_646_7_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, i32 0, i32 %8)
  %11 = call { double, double } @fused_add_add_416_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, i32 %7, i32 %8)
  %12 = extractvalue { double, double } %10, 0
  %13 = extractvalue { double, double } %10, 1
  %14 = extractvalue { double, double } %11, 0
  %15 = extractvalue { double, double } %11, 1
  %16 = fmul double %12, %14
  %17 = fmul double %13, %15
  %18 = fmul double %13, %14
  %19 = fmul double %12, %15
  %20 = fsub double %16, %17
  %21 = fadd double %18, %19
  %22 = insertvalue { double, double } poison, double %20, 0
  %23 = insertvalue { double, double } %22, double %21, 1
  ret { double, double } %23
}

define internal { double, double } @fused_add_scatter_add_646_7_clone_1(ptr noalias %0, ptr noalias %1, ptr noalias %2, ptr noalias %3, ptr noalias %4, ptr noalias %5, ptr noalias %6, i32 %7, i32 %8) {
  %10 = icmp sge i32 %7, 2
  br i1 %10, label %11, label %18

11:                                               ; preds = %9
  %12 = call double @fused_add_dynamic_update_slice_225_8_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, i32 2, i32 %8)
  %13 = call double @fused_add_scatter_add_688_9_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, i32 2, i32 %8)
  %14 = fadd double %12, %13
  %15 = call double @fused_add_dynamic_update_slice_228_8_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, i32 2, i32 %8)
  %16 = call double @fused_add_scatter_add_689_3_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, i32 2, i32 %8)
  %17 = fadd double %15, %16
  br label %21

18:                                               ; preds = %9
  %19 = call double @fused_add_dynamic_update_slice_225_8_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, i32 %7, i32 %8)
  %20 = call double @fused_add_dynamic_update_slice_228_8_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, i32 %7, i32 %8)
  br label %21

21:                                               ; preds = %11, %18
  %22 = phi double [ %19, %18 ], [ %14, %11 ]
  %23 = phi double [ %20, %18 ], [ %17, %11 ]
  br label %24

24:                                               ; preds = %21
  %25 = insertvalue { double, double } poison, double %22, 0
  %26 = insertvalue { double, double } %25, double %23, 1
  ret { double, double } %26
}

define internal double @fused_add_dynamic_update_slice_228_8_clone_1(ptr noalias %0, ptr noalias %1, ptr noalias %2, ptr noalias %3, ptr noalias %4, ptr noalias %5, ptr noalias %6, i32 %7, i32 %8) {
  %10 = icmp sge i32 %7, 1
  %11 = icmp slt i32 %7, 2
  %12 = and i1 %10, %11
  br i1 %12, label %13, label %17

13:                                               ; preds = %9
  %14 = call double @fused_add_dynamic_update_slice_227_9_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, i32 1, i32 %8)
  %15 = call double @fused_add_scatter_add_689_3_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, i32 1, i32 %8)
  %16 = fadd double %14, %15
  br label %19

17:                                               ; preds = %9
  %18 = call double @fused_add_dynamic_update_slice_227_9_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, i32 %7, i32 %8)
  br label %19

19:                                               ; preds = %13, %17
  %20 = phi double [ %18, %17 ], [ %16, %13 ]
  br label %21

21:                                               ; preds = %19
  ret double %20
}

define internal double @fused_add_dynamic_update_slice_227_9_clone_1(ptr noalias %0, ptr noalias %1, ptr noalias %2, ptr noalias %3, ptr noalias %4, ptr noalias %5, ptr noalias %6, i32 %7, i32 %8) {
  %10 = icmp sge i32 %7, 1
  %11 = icmp slt i32 %7, 2
  %12 = and i1 %10, %11
  %13 = sub i32 %7, 1
  br i1 %12, label %14, label %17

14:                                               ; preds = %9
  %15 = call { double, double } @fused_add_mul_384_3_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, i32 %13, i32 %8)
  %16 = extractvalue { double, double } %15, 1
  br label %25

17:                                               ; preds = %9
  %18 = icmp eq i32 %7, 0
  br i1 %18, label %19, label %21

19:                                               ; preds = %17
  %20 = call double @fused_add_scatter_add_689_3_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, i32 %7, i32 %8)
  br label %22

21:                                               ; preds = %17
  br label %22

22:                                               ; preds = %19, %21
  %23 = phi double [ 0.000000e+00, %21 ], [ %20, %19 ]
  br label %24

24:                                               ; preds = %22
  br label %25

25:                                               ; preds = %14, %24
  %26 = phi double [ %23, %24 ], [ %16, %14 ]
  br label %27

27:                                               ; preds = %25
  ret double %26
}

define internal double @fused_add_scatter_add_689_3_clone_1(ptr noalias %0, ptr noalias %1, ptr noalias %2, ptr noalias %3, ptr noalias %4, ptr noalias %5, ptr noalias %6, i32 %7, i32 %8) {
  %10 = mul i32 %7, 64
  %11 = add i32 %10, %8
  %12 = getelementptr inbounds [192 x { double, double }], ptr %5, i32 0, i32 %11
  %13 = load { double, double }, ptr %12, align 8, !invariant.load !3
  %14 = extractvalue { double, double } %13, 1
  ret double %14
}

define internal double @fused_add_dynamic_update_slice_225_8_clone_1(ptr noalias %0, ptr noalias %1, ptr noalias %2, ptr noalias %3, ptr noalias %4, ptr noalias %5, ptr noalias %6, i32 %7, i32 %8) {
  %10 = icmp sge i32 %7, 1
  %11 = icmp slt i32 %7, 2
  %12 = and i1 %10, %11
  br i1 %12, label %13, label %17

13:                                               ; preds = %9
  %14 = call double @fused_add_dynamic_update_slice_224_9_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, i32 1, i32 %8)
  %15 = call double @fused_add_scatter_add_688_9_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, i32 1, i32 %8)
  %16 = fadd double %14, %15
  br label %19

17:                                               ; preds = %9
  %18 = call double @fused_add_dynamic_update_slice_224_9_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, i32 %7, i32 %8)
  br label %19

19:                                               ; preds = %13, %17
  %20 = phi double [ %18, %17 ], [ %16, %13 ]
  br label %21

21:                                               ; preds = %19
  ret double %20
}

define internal double @fused_add_dynamic_update_slice_224_9_clone_1(ptr noalias %0, ptr noalias %1, ptr noalias %2, ptr noalias %3, ptr noalias %4, ptr noalias %5, ptr noalias %6, i32 %7, i32 %8) {
  %10 = icmp sge i32 %7, 1
  %11 = icmp slt i32 %7, 2
  %12 = and i1 %10, %11
  %13 = sub i32 %7, 1
  br i1 %12, label %14, label %17

14:                                               ; preds = %9
  %15 = call { double, double } @fused_add_mul_384_3_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, i32 %13, i32 %8)
  %16 = extractvalue { double, double } %15, 0
  br label %25

17:                                               ; preds = %9
  %18 = icmp eq i32 %7, 0
  br i1 %18, label %19, label %21

19:                                               ; preds = %17
  %20 = call double @fused_add_scatter_add_688_9_clone_1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, i32 %7, i32 %8)
  br label %22

21:                                               ; preds = %17
  br label %22

22:                                               ; preds = %19, %21
  %23 = phi double [ 0.000000e+00, %21 ], [ %20, %19 ]
  br label %24

24:                                               ; preds = %22
  br label %25

25:                                               ; preds = %14, %24
  %26 = phi double [ %23, %24 ], [ %16, %14 ]
  br label %27

27:                                               ; preds = %25
  ret double %26
}

define internal { double, double } @fused_add_mul_384_3_clone_1(ptr noalias %0, ptr noalias %1, ptr noalias %2, ptr noalias %3, ptr noalias %4, ptr noalias %5, ptr noalias %6, i32 %7, i32 %8) {
  %10 = getelementptr inbounds [64 x { double, double }], ptr %6, i32 0, i32 %8
  %11 = load { double, double }, ptr %10, align 8, !invariant.load !3
  %12 = extractvalue { double, double } %11, 0
  %13 = extractvalue { double, double } %11, 1
  %14 = fmul double %12, -0.000000e+00
  %15 = fmul double %13, -1.000000e+00
  %16 = fmul double %13, -0.000000e+00
  %17 = fmul double %12, -1.000000e+00
  %18 = fsub double %14, %15
  %19 = fadd double %16, %17
  %20 = insertvalue { double, double } poison, double %18, 0
  %21 = insertvalue { double, double } %20, double %19, 1
  ret { double, double } %21
}

define internal double @fused_add_scatter_add_688_9_clone_1(ptr noalias %0, ptr noalias %1, ptr noalias %2, ptr noalias %3, ptr noalias %4, ptr noalias %5, ptr noalias %6, i32 %7, i32 %8) {
  %10 = mul i32 %7, 64
  %11 = add i32 %10, %8
  %12 = getelementptr inbounds [192 x { double, double }], ptr %5, i32 0, i32 %11
  %13 = load { double, double }, ptr %12, align 8, !invariant.load !3
  %14 = extractvalue { double, double } %13, 0
  ret double %14
}

define internal { double, double } @fused_add_add_416_1(ptr noalias %0, ptr noalias %1, ptr noalias %2, ptr noalias %3, ptr noalias %4, ptr noalias %5, ptr noalias %6, i32 %7, i32 %8) {
  %10 = mul i32 %7, 64
  %11 = add i32 %10, %8
  %12 = getelementptr inbounds [4096 x { double, double }], ptr %0, i32 0, i32 %11
  %13 = load { double, double }, ptr %12, align 8, !invariant.load !3
  %14 = mul i32 %8, 64
  %15 = add i32 %14, %7
  %16 = getelementptr inbounds [4096 x { double, double }], ptr %1, i32 0, i32 %15
  %17 = load { double, double }, ptr %16, align 8, !invariant.load !3
  %18 = extractvalue { double, double } %13, 0
  %19 = extractvalue { double, double } %17, 0
  %20 = fadd double %18, %19
  %21 = extractvalue { double, double } %13, 1
  %22 = extractvalue { double, double } %17, 1
  %23 = fadd double %21, %22
  %24 = insertvalue { double, double } poison, double %20, 0
  %25 = insertvalue { double, double } %24, double %23, 1
  ret { double, double } %25
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
