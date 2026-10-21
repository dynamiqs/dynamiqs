; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @loop_slice_fusion(ptr noalias align 256 dereferenceable(8) %0, ptr noalias align 16 dereferenceable(8) %1, ptr noalias align 256 dereferenceable(4) %2, ptr noalias align 256 dereferenceable(4) %3) #0 {
  %5 = call i32 @fused_slice_stack_2_9(ptr %0, ptr %1, i32 0, i32 1)
  %6 = call i32 @fused_slice_stack_2_9(ptr %0, ptr %1, i32 1, i32 1)
  %7 = getelementptr inbounds [1 x i32], ptr %2, i32 0, i32 0
  store i32 %5, ptr %7, align 4
  %8 = getelementptr inbounds [1 x i32], ptr %3, i32 0, i32 0
  store i32 %6, ptr %8, align 4
  ret void
}

define internal i32 @fused_slice_stack_2_9(ptr noalias %0, ptr noalias %1, i32 %2, i32 %3) {
  %5 = icmp ult i32 %3, 1
  br i1 %5, label %6, label %9

6:                                                ; preds = %4
  %7 = getelementptr inbounds [2 x i32], ptr %0, i32 0, i32 %2
  %8 = load i32, ptr %7, align 4, !invariant.load !1
  br label %135

9:                                                ; preds = %4
  %10 = sext i32 %2 to i64
  %11 = lshr i64 %10, 32
  %12 = trunc i64 %11 to i32
  %13 = getelementptr inbounds [2 x i32], ptr %1, i32 0, i32 1
  %14 = load i32, ptr %13, align 4, !invariant.load !1
  %15 = add i32 %2, %14
  %16 = shl i32 %15, 13
  %17 = lshr i32 %15, 19
  %18 = getelementptr inbounds [2 x i32], ptr %1, i32 0, i32 0
  %19 = load i32, ptr %18, align 4, !invariant.load !1
  %20 = add i32 %12, %19
  %21 = or i32 %16, %17
  %22 = add i32 %20, %15
  %23 = xor i32 %22, %21
  %24 = shl i32 %23, 15
  %25 = lshr i32 %23, 17
  %26 = or i32 %24, %25
  %27 = add i32 %22, %23
  %28 = xor i32 %27, %26
  %29 = shl i32 %28, 26
  %30 = lshr i32 %28, 6
  %31 = add i32 %27, %28
  %32 = or i32 %29, %30
  %33 = xor i32 %31, %32
  %34 = shl i32 %33, 6
  %35 = lshr i32 %33, 26
  %36 = or i32 %34, %35
  %37 = add i32 %31, %33
  %38 = xor i32 %37, %36
  %39 = xor i32 %19, %14
  %40 = xor i32 %39, 466688986
  %41 = add i32 %38, %40
  %42 = add i32 %41, 1
  %43 = shl i32 %42, 17
  %44 = lshr i32 %42, 15
  %45 = add i32 %37, %14
  %46 = or i32 %43, %44
  %47 = add i32 %45, %42
  %48 = xor i32 %47, %46
  %49 = shl i32 %48, 29
  %50 = lshr i32 %48, 3
  %51 = or i32 %49, %50
  %52 = add i32 %47, %48
  %53 = xor i32 %52, %51
  %54 = shl i32 %53, 16
  %55 = lshr i32 %53, 16
  %56 = add i32 %52, %53
  %57 = or i32 %54, %55
  %58 = xor i32 %56, %57
  %59 = shl i32 %58, 24
  %60 = lshr i32 %58, 8
  %61 = or i32 %59, %60
  %62 = add i32 %56, %58
  %63 = xor i32 %62, %61
  %64 = add i32 %63, %19
  %65 = add i32 %64, 2
  %66 = shl i32 %65, 13
  %67 = lshr i32 %65, 19
  %68 = add i32 %62, %40
  %69 = or i32 %66, %67
  %70 = add i32 %68, %65
  %71 = xor i32 %70, %69
  %72 = shl i32 %71, 15
  %73 = lshr i32 %71, 17
  %74 = or i32 %72, %73
  %75 = add i32 %70, %71
  %76 = xor i32 %75, %74
  %77 = shl i32 %76, 26
  %78 = lshr i32 %76, 6
  %79 = add i32 %75, %76
  %80 = or i32 %77, %78
  %81 = xor i32 %79, %80
  %82 = shl i32 %81, 6
  %83 = lshr i32 %81, 26
  %84 = or i32 %82, %83
  %85 = add i32 %79, %81
  %86 = xor i32 %85, %84
  %87 = add i32 %86, %14
  %88 = add i32 %87, 3
  %89 = shl i32 %88, 17
  %90 = lshr i32 %88, 15
  %91 = add i32 %85, %19
  %92 = or i32 %89, %90
  %93 = add i32 %91, %88
  %94 = xor i32 %93, %92
  %95 = shl i32 %94, 29
  %96 = lshr i32 %94, 3
  %97 = or i32 %95, %96
  %98 = add i32 %93, %94
  %99 = xor i32 %98, %97
  %100 = shl i32 %99, 16
  %101 = lshr i32 %99, 16
  %102 = add i32 %98, %99
  %103 = or i32 %100, %101
  %104 = xor i32 %102, %103
  %105 = shl i32 %104, 24
  %106 = lshr i32 %104, 8
  %107 = or i32 %105, %106
  %108 = add i32 %102, %104
  %109 = xor i32 %108, %107
  %110 = add i32 %109, %40
  %111 = add i32 %110, 4
  %112 = shl i32 %111, 13
  %113 = lshr i32 %111, 19
  %114 = add i32 %108, %14
  %115 = or i32 %112, %113
  %116 = add i32 %114, %111
  %117 = xor i32 %116, %115
  %118 = shl i32 %117, 15
  %119 = lshr i32 %117, 17
  %120 = or i32 %118, %119
  %121 = add i32 %116, %117
  %122 = xor i32 %121, %120
  %123 = shl i32 %122, 26
  %124 = lshr i32 %122, 6
  %125 = or i32 %123, %124
  %126 = add i32 %121, %122
  %127 = xor i32 %126, %125
  %128 = shl i32 %127, 6
  %129 = lshr i32 %127, 26
  %130 = add i32 %126, %127
  %131 = or i32 %128, %129
  %132 = xor i32 %130, %131
  %133 = add i32 %132, %19
  %134 = add i32 %133, 5
  br label %135

135:                                              ; preds = %6, %9
  %136 = phi i32 [ %134, %9 ], [ %8, %6 ]
  br label %137

137:                                              ; preds = %135
  ret i32 %136
}

attributes #0 = { "nvvm.reqntid"="1,1,1" }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{}
