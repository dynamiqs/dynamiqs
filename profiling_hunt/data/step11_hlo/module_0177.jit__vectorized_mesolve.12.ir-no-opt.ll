; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@shared_0 = private addrspace(3) global [8 x i8] undef

define ptx_kernel void @input_reduce_fusion_5(ptr noalias align 256 dereferenceable(4096) %0, ptr noalias align 256 dereferenceable(1) %1) #0 {
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !1
  %4 = mul i32 %3, 4
  %5 = getelementptr inbounds [4096 x i8], ptr %0, i32 0, i32 %4
  %6 = load <4 x i8>, ptr %5, align 1, !invariant.load !2
  %7 = extractelement <4 x i8> %6, i64 0
  %8 = call i8 @region_0_2_reduce_and_5_0(i8 1, i8 %7)
  %9 = extractelement <4 x i8> %6, i64 1
  %10 = call i8 @region_0_2_reduce_and_5_0(i8 %8, i8 %9)
  %11 = extractelement <4 x i8> %6, i64 2
  %12 = call i8 @region_0_2_reduce_and_5_0(i8 %10, i8 %11)
  %13 = extractelement <4 x i8> %6, i64 3
  %14 = call i8 @region_0_2_reduce_and_5_0(i8 %12, i8 %13)
  %15 = add i32 %4, 1024
  %16 = getelementptr inbounds [4096 x i8], ptr %0, i32 0, i32 %15
  %17 = load <4 x i8>, ptr %16, align 1, !invariant.load !2
  %18 = extractelement <4 x i8> %17, i64 0
  %19 = call i8 @region_0_2_reduce_and_5_0(i8 %14, i8 %18)
  %20 = extractelement <4 x i8> %17, i64 1
  %21 = call i8 @region_0_2_reduce_and_5_0(i8 %19, i8 %20)
  %22 = extractelement <4 x i8> %17, i64 2
  %23 = call i8 @region_0_2_reduce_and_5_0(i8 %21, i8 %22)
  %24 = extractelement <4 x i8> %17, i64 3
  %25 = call i8 @region_0_2_reduce_and_5_0(i8 %23, i8 %24)
  %26 = add i32 %4, 2048
  %27 = getelementptr inbounds [4096 x i8], ptr %0, i32 0, i32 %26
  %28 = load <4 x i8>, ptr %27, align 1, !invariant.load !2
  %29 = extractelement <4 x i8> %28, i64 0
  %30 = call i8 @region_0_2_reduce_and_5_0(i8 %25, i8 %29)
  %31 = extractelement <4 x i8> %28, i64 1
  %32 = call i8 @region_0_2_reduce_and_5_0(i8 %30, i8 %31)
  %33 = extractelement <4 x i8> %28, i64 2
  %34 = call i8 @region_0_2_reduce_and_5_0(i8 %32, i8 %33)
  %35 = extractelement <4 x i8> %28, i64 3
  %36 = call i8 @region_0_2_reduce_and_5_0(i8 %34, i8 %35)
  %37 = add i32 %4, 3072
  %38 = getelementptr inbounds [4096 x i8], ptr %0, i32 0, i32 %37
  %39 = load <4 x i8>, ptr %38, align 1, !invariant.load !2
  %40 = extractelement <4 x i8> %39, i64 0
  %41 = call i8 @region_0_2_reduce_and_5_0(i8 %36, i8 %40)
  %42 = extractelement <4 x i8> %39, i64 1
  %43 = call i8 @region_0_2_reduce_and_5_0(i8 %41, i8 %42)
  %44 = extractelement <4 x i8> %39, i64 2
  %45 = call i8 @region_0_2_reduce_and_5_0(i8 %43, i8 %44)
  %46 = extractelement <4 x i8> %39, i64 3
  %47 = call i8 @region_0_2_reduce_and_5_0(i8 %45, i8 %46)
  %48 = zext i8 %47 to i32
  %49 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %48, i32 16, i32 31)
  %50 = trunc i32 %49 to i8
  %51 = call i8 @region_0_2_reduce_and_5_0(i8 %47, i8 %50)
  %52 = zext i8 %51 to i32
  %53 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %52, i32 8, i32 31)
  %54 = trunc i32 %53 to i8
  %55 = call i8 @region_0_2_reduce_and_5_0(i8 %51, i8 %54)
  %56 = zext i8 %55 to i32
  %57 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %56, i32 4, i32 31)
  %58 = trunc i32 %57 to i8
  %59 = call i8 @region_0_2_reduce_and_5_0(i8 %55, i8 %58)
  %60 = zext i8 %59 to i32
  %61 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %60, i32 2, i32 31)
  %62 = trunc i32 %61 to i8
  %63 = call i8 @region_0_2_reduce_and_5_0(i8 %59, i8 %62)
  %64 = zext i8 %63 to i32
  %65 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %64, i32 1, i32 31)
  %66 = trunc i32 %65 to i8
  %67 = call i8 @region_0_2_reduce_and_5_0(i8 %63, i8 %66)
  %68 = urem i32 %3, 32
  %69 = icmp eq i32 %68, 0
  br i1 %69, label %70, label %73

70:                                               ; preds = %2
  %71 = udiv i32 %3, 32
  %72 = getelementptr inbounds [8 x i8], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %71
  store i8 %67, ptr %72, align 1
  br label %73

73:                                               ; preds = %70, %2
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %74 = icmp sle i32 %3, 31
  br i1 %74, label %75, label %110

75:                                               ; preds = %73
  %76 = icmp sle i32 %3, 7
  br i1 %76, label %77, label %80

77:                                               ; preds = %75
  %78 = getelementptr inbounds [8 x i8], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %3
  %79 = load i8, ptr %78, align 1
  br label %81

80:                                               ; preds = %75
  br label %81

81:                                               ; preds = %77, %80
  %82 = phi i8 [ 1, %80 ], [ %79, %77 ]
  br label %83

83:                                               ; preds = %81
  %84 = zext i8 %82 to i32
  %85 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %84, i32 16, i32 31)
  %86 = trunc i32 %85 to i8
  %87 = call i8 @region_0_2_reduce_and_5_0(i8 %82, i8 %86)
  %88 = zext i8 %87 to i32
  %89 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %88, i32 8, i32 31)
  %90 = trunc i32 %89 to i8
  %91 = call i8 @region_0_2_reduce_and_5_0(i8 %87, i8 %90)
  %92 = zext i8 %91 to i32
  %93 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %92, i32 4, i32 31)
  %94 = trunc i32 %93 to i8
  %95 = call i8 @region_0_2_reduce_and_5_0(i8 %91, i8 %94)
  %96 = zext i8 %95 to i32
  %97 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %96, i32 2, i32 31)
  %98 = trunc i32 %97 to i8
  %99 = call i8 @region_0_2_reduce_and_5_0(i8 %95, i8 %98)
  %100 = zext i8 %99 to i32
  %101 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %100, i32 1, i32 31)
  %102 = trunc i32 %101 to i8
  %103 = call i8 @region_0_2_reduce_and_5_0(i8 %99, i8 %102)
  %104 = icmp eq i8 %103, 0
  %105 = zext i1 %104 to i8
  %106 = icmp eq i32 %3, 0
  br i1 %106, label %107, label %109

107:                                              ; preds = %83
  %108 = getelementptr inbounds [1 x i8], ptr %1, i32 0, i32 0
  store i8 %105, ptr %108, align 1
  br label %109

109:                                              ; preds = %107, %83
  br label %110

110:                                              ; preds = %109, %73
  ret void
}

define internal i8 @region_0_2_reduce_and_5_0(i8 %0, i8 %1) {
  %3 = and i8 %0, %1
  ret i8 %3
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: convergent nocallback nounwind memory(inaccessiblemem: readwrite)
declare i32 @llvm.nvvm.shfl.sync.down.i32(i32, i32, i32, i32) #2

; Function Attrs: convergent nocallback nounwind
declare void @llvm.nvvm.barrier.cta.sync.aligned.all(i32) #3

attributes #0 = { "nvvm.reqntid"="256,1,1" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent nocallback nounwind memory(inaccessiblemem: readwrite) }
attributes #3 = { convergent nocallback nounwind }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 0, i32 256}
!2 = !{}
