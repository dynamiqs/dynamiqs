; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@shared_0 = private unnamed_addr addrspace(3) global [8 x i8] undef

; Function Attrs: norecurse nounwind
define ptx_kernel void @input_reduce_fusion_3(ptr noalias nofree readonly align 256 captures(none) dereferenceable(4096) %0, ptr noalias nofree writeonly align 256 captures(none) dereferenceable(1) %1) local_unnamed_addr #0 {
  %3 = addrspacecast ptr %0 to ptr addrspace(1)
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %6 = shl nuw nsw i32 %5, 2
  %7 = zext nneg i32 %6 to i64
  %8 = getelementptr inbounds i8, ptr addrspace(1) %3, i64 %7
  %9 = load <4 x i8>, ptr addrspace(1) %8, align 4, !invariant.load !3
  %10 = extractelement <4 x i8> %9, i32 0
  %11 = extractelement <4 x i8> %9, i32 1
  %12 = extractelement <4 x i8> %9, i32 2
  %13 = extractelement <4 x i8> %9, i32 3
  %14 = and i8 %10, 1
  %15 = and i8 %14, %11
  %16 = and i8 %15, %12
  %17 = and i8 %16, %13
  %18 = getelementptr inbounds i8, ptr addrspace(1) %8, i64 1024
  %19 = load <4 x i8>, ptr addrspace(1) %18, align 4, !invariant.load !3
  %20 = extractelement <4 x i8> %19, i32 0
  %21 = extractelement <4 x i8> %19, i32 1
  %22 = extractelement <4 x i8> %19, i32 2
  %23 = extractelement <4 x i8> %19, i32 3
  %24 = and i8 %17, %20
  %25 = and i8 %24, %21
  %26 = and i8 %25, %22
  %27 = and i8 %26, %23
  %28 = getelementptr inbounds i8, ptr addrspace(1) %8, i64 2048
  %29 = load <4 x i8>, ptr addrspace(1) %28, align 4, !invariant.load !3
  %30 = extractelement <4 x i8> %29, i32 0
  %31 = extractelement <4 x i8> %29, i32 1
  %32 = extractelement <4 x i8> %29, i32 2
  %33 = extractelement <4 x i8> %29, i32 3
  %34 = and i8 %27, %30
  %35 = and i8 %34, %31
  %36 = and i8 %35, %32
  %37 = and i8 %36, %33
  %38 = getelementptr inbounds i8, ptr addrspace(1) %8, i64 3072
  %39 = load <4 x i8>, ptr addrspace(1) %38, align 4, !invariant.load !3
  %40 = extractelement <4 x i8> %39, i32 0
  %41 = extractelement <4 x i8> %39, i32 1
  %42 = extractelement <4 x i8> %39, i32 2
  %43 = extractelement <4 x i8> %39, i32 3
  %44 = and i8 %37, %40
  %45 = and i8 %44, %41
  %46 = and i8 %45, %42
  %47 = and i8 %46, %43
  %48 = zext nneg i8 %47 to i32
  %49 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %48, i32 16, i32 31)
  %50 = trunc i32 %49 to i8
  %51 = and i8 %47, %50
  %52 = zext nneg i8 %51 to i32
  %53 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %52, i32 8, i32 31)
  %54 = trunc i32 %53 to i8
  %55 = and i8 %51, %54
  %56 = zext nneg i8 %55 to i32
  %57 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %56, i32 4, i32 31)
  %58 = trunc i32 %57 to i8
  %59 = and i8 %55, %58
  %60 = zext nneg i8 %59 to i32
  %61 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %60, i32 2, i32 31)
  %62 = trunc i32 %61 to i8
  %63 = and i8 %59, %62
  %64 = zext nneg i8 %63 to i32
  %65 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %64, i32 1, i32 31)
  %66 = and i32 %5, 31
  %67 = icmp eq i32 %66, 0
  %68 = lshr exact i32 %5, 5
  %69 = zext nneg i32 %68 to i64
  %70 = getelementptr inbounds i8, ptr addrspace(3) @shared_0, i64 %69
  %71 = trunc i32 %65 to i8
  %72 = and i8 %63, %71
  br i1 %67, label %73, label %74

73:                                               ; preds = %2
  store i8 %72, ptr addrspace(3) %70, align 1
  br label %74

74:                                               ; preds = %73, %2
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %75 = icmp samesign ult i32 %5, 32
  br i1 %75, label %76, label %108

76:                                               ; preds = %74
  %77 = icmp samesign ult i32 %5, 8
  %78 = zext nneg i32 %5 to i64
  %79 = getelementptr inbounds i8, ptr addrspace(3) @shared_0, i64 %78
  br i1 %77, label %80, label %82

80:                                               ; preds = %76
  %81 = load i8, ptr addrspace(3) %79, align 1
  br label %82

82:                                               ; preds = %80, %76
  %83 = phi i8 [ %81, %80 ], [ 1, %76 ]
  %84 = zext i8 %83 to i32
  %85 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %84, i32 16, i32 31)
  %86 = trunc i32 %85 to i8
  %87 = and i8 %83, %86
  %88 = zext i8 %87 to i32
  %89 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %88, i32 8, i32 31)
  %90 = trunc i32 %89 to i8
  %91 = and i8 %87, %90
  %92 = zext i8 %91 to i32
  %93 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %92, i32 4, i32 31)
  %94 = trunc i32 %93 to i8
  %95 = and i8 %91, %94
  %96 = zext i8 %95 to i32
  %97 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %96, i32 2, i32 31)
  %98 = trunc i32 %97 to i8
  %99 = and i8 %95, %98
  %100 = zext i8 %99 to i32
  %101 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %100, i32 1, i32 31)
  %102 = icmp eq i32 %5, 0
  %103 = trunc i32 %101 to i8
  %104 = and i8 %99, %103
  %105 = icmp eq i8 %104, 0
  %106 = zext i1 %105 to i8
  br i1 %102, label %107, label %108

107:                                              ; preds = %82
  store i8 %106, ptr addrspace(1) %4, align 256
  br label %108

108:                                              ; preds = %82, %107, %74
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
