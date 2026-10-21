; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @loop_multiply_fusion(ptr noalias align 16 dereferenceable(2048) %0, ptr noalias align 16 dereferenceable(1024) %1, ptr noalias align 256 dereferenceable(3072) %2) #0 {
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !1
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %6 = mul i32 %4, 128
  %7 = add i32 %6, %5
  %8 = icmp sle i32 %7, 191
  br i1 %8, label %9, label %77

9:                                                ; preds = %3
  %10 = mul i32 %4, 2
  %11 = udiv i32 %5, 64
  %12 = add i32 %10, %11
  %13 = urem i32 %12, 3
  %14 = urem i32 %5, 64
  %15 = icmp sge i32 %13, 2
  %16 = sub i32 %13, 2
  br i1 %15, label %17, label %59

17:                                               ; preds = %9
  %18 = call double @fused_multiply_dynamic_update_slice_219_1(ptr %0, ptr %1, i32 2, i32 %14)
  %19 = icmp eq i32 %16, 0
  %20 = icmp sge i32 %14, 2
  %21 = and i1 %19, %20
  br i1 %21, label %22, label %35

22:                                               ; preds = %17
  %23 = add i32 %14, -2
  %24 = call { double, double } @fused_multiply_complex_32_28(ptr %0, ptr %1, i32 %16, i32 %23)
  %25 = add i32 %14, 64
  %26 = getelementptr inbounds [128 x { double, double }], ptr %0, i32 0, i32 %25
  %27 = load { double, double }, ptr %26, align 8, !invariant.load !3
  %28 = extractvalue { double, double } %24, 0
  %29 = extractvalue { double, double } %24, 1
  %30 = extractvalue { double, double } %27, 0
  %31 = extractvalue { double, double } %27, 1
  %32 = fmul double %28, %30
  %33 = fmul double %29, %31
  %34 = fsub double %32, %33
  br label %36

35:                                               ; preds = %17
  br label %36

36:                                               ; preds = %22, %35
  %37 = phi double [ 0.000000e+00, %35 ], [ %34, %22 ]
  br label %38

38:                                               ; preds = %36
  %39 = fadd double %18, %37
  %40 = call double @fused_multiply_dynamic_update_slice_222_1(ptr %0, ptr %1, i32 2, i32 %14)
  br i1 %21, label %41, label %54

41:                                               ; preds = %38
  %42 = add i32 %14, -2
  %43 = call { double, double } @fused_multiply_complex_32_28(ptr %0, ptr %1, i32 %16, i32 %42)
  %44 = add i32 %14, 64
  %45 = getelementptr inbounds [128 x { double, double }], ptr %0, i32 0, i32 %44
  %46 = load { double, double }, ptr %45, align 8, !invariant.load !3
  %47 = extractvalue { double, double } %43, 0
  %48 = extractvalue { double, double } %43, 1
  %49 = extractvalue { double, double } %46, 0
  %50 = extractvalue { double, double } %46, 1
  %51 = fmul double %48, %49
  %52 = fmul double %47, %50
  %53 = fadd double %51, %52
  br label %55

54:                                               ; preds = %38
  br label %55

55:                                               ; preds = %41, %54
  %56 = phi double [ 0.000000e+00, %54 ], [ %53, %41 ]
  br label %57

57:                                               ; preds = %55
  %58 = fadd double %40, %56
  br label %62

59:                                               ; preds = %9
  %60 = call double @fused_multiply_dynamic_update_slice_219_1(ptr %0, ptr %1, i32 %13, i32 %14)
  %61 = call double @fused_multiply_dynamic_update_slice_222_1(ptr %0, ptr %1, i32 %13, i32 %14)
  br label %62

62:                                               ; preds = %57, %59
  %63 = phi double [ %60, %59 ], [ %39, %57 ]
  %64 = phi double [ %61, %59 ], [ %58, %57 ]
  br label %65

65:                                               ; preds = %62
  %66 = fmul double %63, -5.000000e-01
  %67 = fmul double %64, 0.000000e+00
  %68 = fmul double %64, -5.000000e-01
  %69 = fmul double %63, 0.000000e+00
  %70 = fsub double %66, %67
  %71 = fadd double %68, %69
  %72 = insertvalue { double, double } poison, double %70, 0
  %73 = insertvalue { double, double } %72, double %71, 1
  %74 = mul i32 %13, 64
  %75 = add i32 %74, %14
  %76 = getelementptr inbounds [192 x { double, double }], ptr %2, i32 0, i32 %75
  store { double, double } %73, ptr %76, align 8
  br label %77

77:                                               ; preds = %65, %3
  ret void
}

define internal double @fused_multiply_dynamic_update_slice_222_1(ptr noalias %0, ptr noalias %1, i32 %2, i32 %3) {
  %5 = icmp sge i32 %2, 1
  %6 = icmp slt i32 %2, 2
  %7 = and i1 %5, %6
  %8 = sub i32 %2, 1
  br i1 %7, label %9, label %23

9:                                                ; preds = %4
  %10 = call double @fused_multiply_dynamic_update_slice_221_3(ptr %0, ptr %1, i32 1, i32 %3)
  %11 = icmp eq i32 %8, 0
  %12 = icmp sge i32 %3, 1
  %13 = and i1 %11, %12
  br i1 %13, label %14, label %18

14:                                               ; preds = %9
  %15 = add i32 %3, -1
  %16 = call { double, double } @fused_multiply_mul_414_3(ptr %0, ptr %1, i32 %8, i32 %15)
  %17 = extractvalue { double, double } %16, 1
  br label %19

18:                                               ; preds = %9
  br label %19

19:                                               ; preds = %14, %18
  %20 = phi double [ 0.000000e+00, %18 ], [ %17, %14 ]
  br label %21

21:                                               ; preds = %19
  %22 = fadd double %10, %20
  br label %25

23:                                               ; preds = %4
  %24 = call double @fused_multiply_dynamic_update_slice_221_3(ptr %0, ptr %1, i32 %2, i32 %3)
  br label %25

25:                                               ; preds = %21, %23
  %26 = phi double [ %24, %23 ], [ %22, %21 ]
  br label %27

27:                                               ; preds = %25
  ret double %26
}

define internal double @fused_multiply_dynamic_update_slice_221_3(ptr noalias %0, ptr noalias %1, i32 %2, i32 %3) {
  %5 = icmp sge i32 %2, 1
  %6 = icmp slt i32 %2, 2
  %7 = and i1 %5, %6
  %8 = sub i32 %2, 1
  br i1 %7, label %9, label %26

9:                                                ; preds = %4
  %10 = icmp sge i32 %8, 0
  %11 = icmp slt i32 %8, 1
  %12 = and i1 %10, %11
  %13 = icmp sge i32 %3, 2
  %14 = and i1 %12, %13
  %15 = sub i32 %3, 2
  br i1 %14, label %16, label %21

16:                                               ; preds = %9
  %17 = call double @fused_multiply_scatter_add_694_2(ptr %0, ptr %1, i32 %8, i32 %3)
  %18 = call { double, double } @fused_multiply_mul_412_3(ptr %0, ptr %1, i32 %8, i32 %15)
  %19 = extractvalue { double, double } %18, 1
  %20 = fadd double %17, %19
  br label %23

21:                                               ; preds = %9
  %22 = call double @fused_multiply_scatter_add_694_2(ptr %0, ptr %1, i32 %8, i32 %3)
  br label %23

23:                                               ; preds = %16, %21
  %24 = phi double [ %22, %21 ], [ %20, %16 ]
  br label %25

25:                                               ; preds = %23
  br label %35

26:                                               ; preds = %4
  %27 = icmp eq i32 %2, 0
  br i1 %27, label %28, label %31

28:                                               ; preds = %26
  %29 = call { double, double } @fused_multiply_mul_410_7(ptr %0, ptr %1, i32 %2, i32 %3)
  %30 = extractvalue { double, double } %29, 1
  br label %32

31:                                               ; preds = %26
  br label %32

32:                                               ; preds = %28, %31
  %33 = phi double [ 0.000000e+00, %31 ], [ %30, %28 ]
  br label %34

34:                                               ; preds = %32
  br label %35

35:                                               ; preds = %25, %34
  %36 = phi double [ %33, %34 ], [ %24, %25 ]
  br label %37

37:                                               ; preds = %35
  ret double %36
}

define internal double @fused_multiply_scatter_add_694_2(ptr noalias %0, ptr noalias %1, i32 %2, i32 %3) {
  %5 = call { double, double } @fused_multiply_mul_411_3(ptr %0, ptr %1, i32 %2, i32 %3)
  %6 = extractvalue { double, double } %5, 1
  ret double %6
}

define internal double @fused_multiply_dynamic_update_slice_219_1(ptr noalias %0, ptr noalias %1, i32 %2, i32 %3) {
  %5 = icmp sge i32 %2, 1
  %6 = icmp slt i32 %2, 2
  %7 = and i1 %5, %6
  %8 = sub i32 %2, 1
  br i1 %7, label %9, label %23

9:                                                ; preds = %4
  %10 = call double @fused_multiply_dynamic_update_slice_218_3(ptr %0, ptr %1, i32 1, i32 %3)
  %11 = icmp eq i32 %8, 0
  %12 = icmp sge i32 %3, 1
  %13 = and i1 %11, %12
  br i1 %13, label %14, label %18

14:                                               ; preds = %9
  %15 = add i32 %3, -1
  %16 = call { double, double } @fused_multiply_mul_414_3(ptr %0, ptr %1, i32 %8, i32 %15)
  %17 = extractvalue { double, double } %16, 0
  br label %19

18:                                               ; preds = %9
  br label %19

19:                                               ; preds = %14, %18
  %20 = phi double [ 0.000000e+00, %18 ], [ %17, %14 ]
  br label %21

21:                                               ; preds = %19
  %22 = fadd double %10, %20
  br label %25

23:                                               ; preds = %4
  %24 = call double @fused_multiply_dynamic_update_slice_218_3(ptr %0, ptr %1, i32 %2, i32 %3)
  br label %25

25:                                               ; preds = %21, %23
  %26 = phi double [ %24, %23 ], [ %22, %21 ]
  br label %27

27:                                               ; preds = %25
  ret double %26
}

define internal { double, double } @fused_multiply_mul_414_3(ptr noalias %0, ptr noalias %1, i32 %2, i32 %3) {
  %5 = add i32 %3, 1
  %6 = getelementptr inbounds [64 x { double, double }], ptr %1, i32 0, i32 %5
  %7 = load { double, double }, ptr %6, align 8, !invariant.load !3
  %8 = extractvalue { double, double } %7, 1
  %9 = extractvalue { double, double } %7, 0
  %10 = fneg double %8
  %11 = fmul double %9, %9
  %12 = fmul double %10, %8
  %13 = fmul double %10, %9
  %14 = fmul double %9, %8
  %15 = fsub double %11, %12
  %16 = fadd double %13, %14
  %17 = insertvalue { double, double } poison, double %15, 0
  %18 = insertvalue { double, double } %17, double %16, 1
  ret { double, double } %18
}

define internal double @fused_multiply_dynamic_update_slice_218_3(ptr noalias %0, ptr noalias %1, i32 %2, i32 %3) {
  %5 = icmp sge i32 %2, 1
  %6 = icmp slt i32 %2, 2
  %7 = and i1 %5, %6
  %8 = sub i32 %2, 1
  br i1 %7, label %9, label %26

9:                                                ; preds = %4
  %10 = icmp sge i32 %8, 0
  %11 = icmp slt i32 %8, 1
  %12 = and i1 %10, %11
  %13 = icmp sge i32 %3, 2
  %14 = and i1 %12, %13
  %15 = sub i32 %3, 2
  br i1 %14, label %16, label %21

16:                                               ; preds = %9
  %17 = call double @fused_multiply_scatter_add_692_2(ptr %0, ptr %1, i32 %8, i32 %3)
  %18 = call { double, double } @fused_multiply_mul_412_3(ptr %0, ptr %1, i32 %8, i32 %15)
  %19 = extractvalue { double, double } %18, 0
  %20 = fadd double %17, %19
  br label %23

21:                                               ; preds = %9
  %22 = call double @fused_multiply_scatter_add_692_2(ptr %0, ptr %1, i32 %8, i32 %3)
  br label %23

23:                                               ; preds = %16, %21
  %24 = phi double [ %22, %21 ], [ %20, %16 ]
  br label %25

25:                                               ; preds = %23
  br label %35

26:                                               ; preds = %4
  %27 = icmp eq i32 %2, 0
  br i1 %27, label %28, label %31

28:                                               ; preds = %26
  %29 = call { double, double } @fused_multiply_mul_410_7(ptr %0, ptr %1, i32 %2, i32 %3)
  %30 = extractvalue { double, double } %29, 0
  br label %32

31:                                               ; preds = %26
  br label %32

32:                                               ; preds = %28, %31
  %33 = phi double [ 0.000000e+00, %31 ], [ %30, %28 ]
  br label %34

34:                                               ; preds = %32
  br label %35

35:                                               ; preds = %25, %34
  %36 = phi double [ %33, %34 ], [ %24, %25 ]
  br label %37

37:                                               ; preds = %35
  ret double %36
}

define internal { double, double } @fused_multiply_mul_412_3(ptr noalias %0, ptr noalias %1, i32 %2, i32 %3) {
  %5 = call { double, double } @fused_multiply_complex_32_28(ptr %0, ptr %1, i32 1, i32 %3)
  %6 = add i32 %3, 66
  %7 = getelementptr inbounds [128 x { double, double }], ptr %0, i32 0, i32 %6
  %8 = load { double, double }, ptr %7, align 8, !invariant.load !3
  %9 = extractvalue { double, double } %5, 0
  %10 = extractvalue { double, double } %5, 1
  %11 = extractvalue { double, double } %8, 0
  %12 = extractvalue { double, double } %8, 1
  %13 = fmul double %9, %11
  %14 = fmul double %10, %12
  %15 = fmul double %10, %11
  %16 = fmul double %9, %12
  %17 = fsub double %13, %14
  %18 = fadd double %15, %16
  %19 = insertvalue { double, double } poison, double %17, 0
  %20 = insertvalue { double, double } %19, double %18, 1
  ret { double, double } %20
}

define internal double @fused_multiply_scatter_add_692_2(ptr noalias %0, ptr noalias %1, i32 %2, i32 %3) {
  %5 = call { double, double } @fused_multiply_mul_411_3(ptr %0, ptr %1, i32 %2, i32 %3)
  %6 = extractvalue { double, double } %5, 0
  ret double %6
}

define internal { double, double } @fused_multiply_mul_411_3(ptr noalias %0, ptr noalias %1, i32 %2, i32 %3) {
  %5 = call { double, double } @fused_multiply_complex_32_28(ptr %0, ptr %1, i32 %2, i32 %3)
  %6 = mul i32 %2, 64
  %7 = add i32 %6, %3
  %8 = getelementptr inbounds [128 x { double, double }], ptr %0, i32 0, i32 %7
  %9 = load { double, double }, ptr %8, align 8, !invariant.load !3
  %10 = extractvalue { double, double } %5, 0
  %11 = extractvalue { double, double } %5, 1
  %12 = extractvalue { double, double } %9, 0
  %13 = extractvalue { double, double } %9, 1
  %14 = fmul double %10, %12
  %15 = fmul double %11, %13
  %16 = fmul double %11, %12
  %17 = fmul double %10, %13
  %18 = fsub double %14, %15
  %19 = fadd double %16, %17
  %20 = insertvalue { double, double } poison, double %18, 0
  %21 = insertvalue { double, double } %20, double %19, 1
  ret { double, double } %21
}

define internal { double, double } @fused_multiply_mul_410_7(ptr noalias %0, ptr noalias %1, i32 %2, i32 %3) {
  %5 = call { double, double } @fused_multiply_complex_32_28(ptr %0, ptr %1, i32 1, i32 %3)
  %6 = mul i32 %2, 64
  %7 = add i32 %6, %3
  %8 = getelementptr inbounds [128 x { double, double }], ptr %0, i32 0, i32 %7
  %9 = load { double, double }, ptr %8, align 8, !invariant.load !3
  %10 = extractvalue { double, double } %5, 0
  %11 = extractvalue { double, double } %5, 1
  %12 = extractvalue { double, double } %9, 0
  %13 = extractvalue { double, double } %9, 1
  %14 = fmul double %10, %12
  %15 = fmul double %11, %13
  %16 = fmul double %11, %12
  %17 = fmul double %10, %13
  %18 = fsub double %14, %15
  %19 = fadd double %16, %17
  %20 = insertvalue { double, double } poison, double %18, 0
  %21 = insertvalue { double, double } %20, double %19, 1
  ret { double, double } %21
}

define internal { double, double } @fused_multiply_complex_32_28(ptr noalias %0, ptr noalias %1, i32 %2, i32 %3) {
  %5 = icmp sge i32 %2, 1
  %6 = icmp slt i32 %3, 62
  %7 = and i1 %5, %6
  br i1 %7, label %8, label %12

8:                                                ; preds = %4
  %9 = add i32 %3, 66
  %10 = getelementptr inbounds [128 x { double, double }], ptr %0, i32 0, i32 %9
  %11 = load { double, double }, ptr %10, align 8, !invariant.load !3
  br label %23

12:                                               ; preds = %4
  %13 = icmp eq i32 %2, 0
  br i1 %13, label %14, label %19

14:                                               ; preds = %12
  %15 = mul i32 %2, 64
  %16 = add i32 %15, %3
  %17 = getelementptr inbounds [128 x { double, double }], ptr %0, i32 0, i32 %16
  %18 = load { double, double }, ptr %17, align 8, !invariant.load !3
  br label %20

19:                                               ; preds = %12
  br label %20

20:                                               ; preds = %14, %19
  %21 = phi { double, double } [ zeroinitializer, %19 ], [ %18, %14 ]
  br label %22

22:                                               ; preds = %20
  br label %23

23:                                               ; preds = %8, %22
  %24 = phi { double, double } [ %21, %22 ], [ %11, %8 ]
  br label %25

25:                                               ; preds = %23
  %26 = extractvalue { double, double } %24, 1
  %27 = extractvalue { double, double } %24, 0
  %28 = fneg double %26
  %29 = insertvalue { double, double } poison, double %27, 0
  %30 = insertvalue { double, double } %29, double %28, 1
  ret { double, double } %30
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

attributes #0 = { "nvvm.reqntid"="128,1,1" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 0, i32 2}
!2 = !{i32 0, i32 128}
!3 = !{}
