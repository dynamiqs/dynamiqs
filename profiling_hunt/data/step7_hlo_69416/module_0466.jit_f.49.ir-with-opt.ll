; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@shared_0 = private unnamed_addr addrspace(3) global [8 x i8] undef

; Function Attrs: norecurse nounwind
define ptx_kernel void @input_reduce_fusion_1(ptr noalias nofree readonly align 256 captures(none) dereferenceable(65536) %0, ptr noalias nofree writeonly align 256 captures(none) dereferenceable(1) %1) local_unnamed_addr #0 {
  %3 = addrspacecast ptr %0 to ptr addrspace(1)
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %6 = zext nneg i32 %5 to i64
  %7 = getelementptr inbounds [16 x i8], ptr addrspace(1) %3, i64 %6
  %8 = load <2 x double>, ptr addrspace(1) %7, align 16, !invariant.load !3
  %.unpack48 = extractelement <2 x double> %8, i32 0
  %.unpack249 = extractelement <2 x double> %8, i32 1
  %9 = fcmp uno double %.unpack48, %.unpack249
  %10 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 4096
  %11 = load <2 x double>, ptr addrspace(1) %10, align 16, !invariant.load !3
  %.unpack350 = extractelement <2 x double> %11, i32 0
  %.unpack551 = extractelement <2 x double> %11, i32 1
  %12 = fcmp uno double %.unpack350, %.unpack551
  %13 = or i1 %9, %12
  %14 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 8192
  %15 = load <2 x double>, ptr addrspace(1) %14, align 16, !invariant.load !3
  %.unpack652 = extractelement <2 x double> %15, i32 0
  %.unpack853 = extractelement <2 x double> %15, i32 1
  %16 = fcmp uno double %.unpack652, %.unpack853
  %17 = or i1 %13, %16
  %18 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 12288
  %19 = load <2 x double>, ptr addrspace(1) %18, align 16, !invariant.load !3
  %.unpack954 = extractelement <2 x double> %19, i32 0
  %.unpack1155 = extractelement <2 x double> %19, i32 1
  %20 = fcmp uno double %.unpack954, %.unpack1155
  %21 = or i1 %17, %20
  %22 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 16384
  %23 = load <2 x double>, ptr addrspace(1) %22, align 16, !invariant.load !3
  %.unpack1256 = extractelement <2 x double> %23, i32 0
  %.unpack1457 = extractelement <2 x double> %23, i32 1
  %24 = fcmp uno double %.unpack1256, %.unpack1457
  %25 = or i1 %21, %24
  %26 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 20480
  %27 = load <2 x double>, ptr addrspace(1) %26, align 16, !invariant.load !3
  %.unpack1558 = extractelement <2 x double> %27, i32 0
  %.unpack1759 = extractelement <2 x double> %27, i32 1
  %28 = fcmp uno double %.unpack1558, %.unpack1759
  %29 = or i1 %25, %28
  %30 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 24576
  %31 = load <2 x double>, ptr addrspace(1) %30, align 16, !invariant.load !3
  %.unpack1860 = extractelement <2 x double> %31, i32 0
  %.unpack2061 = extractelement <2 x double> %31, i32 1
  %32 = fcmp uno double %.unpack1860, %.unpack2061
  %33 = or i1 %29, %32
  %34 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 28672
  %35 = load <2 x double>, ptr addrspace(1) %34, align 16, !invariant.load !3
  %.unpack2162 = extractelement <2 x double> %35, i32 0
  %.unpack2363 = extractelement <2 x double> %35, i32 1
  %36 = fcmp uno double %.unpack2162, %.unpack2363
  %37 = or i1 %33, %36
  %38 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 32768
  %39 = load <2 x double>, ptr addrspace(1) %38, align 16, !invariant.load !3
  %.unpack2464 = extractelement <2 x double> %39, i32 0
  %.unpack2665 = extractelement <2 x double> %39, i32 1
  %40 = fcmp uno double %.unpack2464, %.unpack2665
  %41 = or i1 %37, %40
  %42 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 36864
  %43 = load <2 x double>, ptr addrspace(1) %42, align 16, !invariant.load !3
  %.unpack2766 = extractelement <2 x double> %43, i32 0
  %.unpack2967 = extractelement <2 x double> %43, i32 1
  %44 = fcmp uno double %.unpack2766, %.unpack2967
  %45 = or i1 %41, %44
  %46 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 40960
  %47 = load <2 x double>, ptr addrspace(1) %46, align 16, !invariant.load !3
  %.unpack3068 = extractelement <2 x double> %47, i32 0
  %.unpack3269 = extractelement <2 x double> %47, i32 1
  %48 = fcmp uno double %.unpack3068, %.unpack3269
  %49 = or i1 %45, %48
  %50 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 45056
  %51 = load <2 x double>, ptr addrspace(1) %50, align 16, !invariant.load !3
  %.unpack3370 = extractelement <2 x double> %51, i32 0
  %.unpack3571 = extractelement <2 x double> %51, i32 1
  %52 = fcmp uno double %.unpack3370, %.unpack3571
  %53 = or i1 %49, %52
  %54 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 49152
  %55 = load <2 x double>, ptr addrspace(1) %54, align 16, !invariant.load !3
  %.unpack3672 = extractelement <2 x double> %55, i32 0
  %.unpack3873 = extractelement <2 x double> %55, i32 1
  %56 = fcmp uno double %.unpack3672, %.unpack3873
  %57 = or i1 %53, %56
  %58 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 53248
  %59 = load <2 x double>, ptr addrspace(1) %58, align 16, !invariant.load !3
  %.unpack3974 = extractelement <2 x double> %59, i32 0
  %.unpack4175 = extractelement <2 x double> %59, i32 1
  %60 = fcmp uno double %.unpack3974, %.unpack4175
  %61 = or i1 %57, %60
  %62 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 57344
  %63 = load <2 x double>, ptr addrspace(1) %62, align 16, !invariant.load !3
  %.unpack4276 = extractelement <2 x double> %63, i32 0
  %.unpack4477 = extractelement <2 x double> %63, i32 1
  %64 = fcmp uno double %.unpack4276, %.unpack4477
  %65 = or i1 %61, %64
  %66 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 61440
  %67 = load <2 x double>, ptr addrspace(1) %66, align 16, !invariant.load !3
  %.unpack4578 = extractelement <2 x double> %67, i32 0
  %.unpack4779 = extractelement <2 x double> %67, i32 1
  %68 = fcmp uno double %.unpack4578, %.unpack4779
  %69 = or i1 %65, %68
  %70 = zext i1 %69 to i8
  %71 = zext i1 %69 to i32
  %72 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %71, i32 16, i32 31)
  %73 = trunc i32 %72 to i8
  %74 = or i8 %70, %73
  %75 = zext i8 %74 to i32
  %76 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %75, i32 8, i32 31)
  %77 = trunc i32 %76 to i8
  %78 = or i8 %74, %77
  %79 = zext i8 %78 to i32
  %80 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %79, i32 4, i32 31)
  %81 = trunc i32 %80 to i8
  %82 = or i8 %78, %81
  %83 = zext i8 %82 to i32
  %84 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %83, i32 2, i32 31)
  %85 = trunc i32 %84 to i8
  %86 = or i8 %82, %85
  %87 = zext i8 %86 to i32
  %88 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %87, i32 1, i32 31)
  %89 = and i32 %5, 31
  %90 = icmp eq i32 %89, 0
  %91 = lshr exact i32 %5, 5
  %92 = zext nneg i32 %91 to i64
  %93 = getelementptr inbounds i8, ptr addrspace(3) @shared_0, i64 %92
  %94 = trunc i32 %88 to i8
  %95 = or i8 %86, %94
  br i1 %90, label %96, label %97

96:                                               ; preds = %2
  store i8 %95, ptr addrspace(3) %93, align 1
  br label %97

97:                                               ; preds = %96, %2
  %98 = trunc i64 %6 to i32
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %99 = icmp samesign ult i32 %98, 32
  br i1 %99, label %100, label %131

100:                                              ; preds = %97
  %101 = trunc i64 %6 to i32
  %102 = icmp samesign ult i32 %101, 8
  %103 = getelementptr inbounds i8, ptr addrspace(3) @shared_0, i64 %6
  br i1 %102, label %104, label %106

104:                                              ; preds = %100
  %105 = load i8, ptr addrspace(3) %103, align 1
  br label %106

106:                                              ; preds = %104, %100
  %107 = phi i8 [ %105, %104 ], [ 0, %100 ]
  %108 = trunc i64 %6 to i32
  %109 = zext i8 %107 to i32
  %110 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %109, i32 16, i32 31)
  %111 = trunc i32 %110 to i8
  %112 = or i8 %107, %111
  %113 = zext i8 %112 to i32
  %114 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %113, i32 8, i32 31)
  %115 = trunc i32 %114 to i8
  %116 = or i8 %112, %115
  %117 = zext i8 %116 to i32
  %118 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %117, i32 4, i32 31)
  %119 = trunc i32 %118 to i8
  %120 = or i8 %116, %119
  %121 = zext i8 %120 to i32
  %122 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %121, i32 2, i32 31)
  %123 = trunc i32 %122 to i8
  %124 = or i8 %120, %123
  %125 = zext i8 %124 to i32
  %126 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %125, i32 1, i32 31)
  %127 = icmp eq i32 %108, 0
  %128 = trunc i32 %126 to i8
  %129 = or i8 %124, %128
  br i1 %127, label %130, label %131

130:                                              ; preds = %106
  store i8 %129, ptr addrspace(1) %4, align 256
  br label %131

131:                                              ; preds = %106, %130, %97
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: convergent nocallback nounwind memory(inaccessiblemem: readwrite)
declare i32 @llvm.nvvm.shfl.sync.down.i32(i32, i32, i32, i32) #2

; Function Attrs: convergent nocallback nounwind
declare void @llvm.nvvm.barrier.cta.sync.aligned.all(i32) #3

attributes #0 = { norecurse nounwind "nvvm.reqntid"="256,1,1" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent nocallback nounwind memory(inaccessiblemem: readwrite) }
attributes #3 = { convergent nocallback nounwind }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{i32 0, i32 256}
!3 = !{}
