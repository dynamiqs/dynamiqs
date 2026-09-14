; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@shared_0 = private unnamed_addr addrspace(3) global [1056 x { double, double }] undef

; Function Attrs: norecurse nounwind
define ptx_kernel void @input_transpose_fusion_1(ptr noalias nofree align 256 captures(none) dereferenceable(65536) %0, ptr noalias nofree readonly align 256 captures(none) dereferenceable(65536) %1, ptr noalias nofree readonly align 256 captures(none) dereferenceable(1) %2, ptr noalias nofree readnone align 256 captures(none) dereferenceable(65536) %3) local_unnamed_addr #0 {
  %5 = addrspacecast ptr %1 to ptr addrspace(1)
  %6 = addrspacecast ptr %2 to ptr addrspace(1)
  %7 = addrspacecast ptr %0 to ptr addrspace(1)
  %8 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %9 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !3
  %10 = and i32 %9, 1
  %11 = shl nuw nsw i32 %10, 5
  %12 = lshr i32 %8, 5
  %13 = shl nuw nsw i32 %12, 6
  %14 = lshr i32 %9, 1
  %15 = shl nuw nsw i32 %14, 11
  %16 = or disjoint i32 %11, %15
  %17 = or disjoint i32 %16, %13
  %18 = and i32 %8, 31
  %19 = or disjoint i32 %17, %18
  %20 = zext nneg i32 %19 to i64
  %21 = getelementptr inbounds [16 x i8], ptr addrspace(1) %5, i64 %20
  %22 = load <2 x double>, ptr addrspace(1) %21, align 16, !invariant.load !4
  %.unpack125 = extractelement <2 x double> %22, i32 0
  %.unpack2126 = extractelement <2 x double> %22, i32 1
  %23 = mul nuw nsw i32 %18, 33
  %24 = add nuw nsw i32 %23, %12
  %25 = zext nneg i32 %24 to i64
  %26 = getelementptr inbounds [16 x i8], ptr addrspace(3) @shared_0, i64 %25
  store double %.unpack125, ptr addrspace(3) %26, align 8
  %.repack3 = getelementptr inbounds i8, ptr addrspace(3) %26, i64 8
  store double %.unpack2126, ptr addrspace(3) %.repack3, align 8
  %27 = getelementptr inbounds i8, ptr addrspace(1) %21, i64 4096
  %28 = load <2 x double>, ptr addrspace(1) %27, align 16, !invariant.load !4
  %.unpack5127 = extractelement <2 x double> %28, i32 0
  %.unpack7128 = extractelement <2 x double> %28, i32 1
  %29 = getelementptr inbounds i8, ptr addrspace(3) %26, i64 64
  store double %.unpack5127, ptr addrspace(3) %29, align 8
  %.repack8 = getelementptr inbounds i8, ptr addrspace(3) %26, i64 72
  store double %.unpack7128, ptr addrspace(3) %.repack8, align 8
  %30 = getelementptr inbounds i8, ptr addrspace(1) %21, i64 8192
  %31 = load <2 x double>, ptr addrspace(1) %30, align 16, !invariant.load !4
  %.unpack10129 = extractelement <2 x double> %31, i32 0
  %.unpack12130 = extractelement <2 x double> %31, i32 1
  %32 = getelementptr inbounds i8, ptr addrspace(3) %26, i64 128
  store double %.unpack10129, ptr addrspace(3) %32, align 8
  %.repack13 = getelementptr inbounds i8, ptr addrspace(3) %26, i64 136
  store double %.unpack12130, ptr addrspace(3) %.repack13, align 8
  %33 = getelementptr inbounds i8, ptr addrspace(1) %21, i64 12288
  %34 = load <2 x double>, ptr addrspace(1) %33, align 16, !invariant.load !4
  %.unpack15131 = extractelement <2 x double> %34, i32 0
  %.unpack17132 = extractelement <2 x double> %34, i32 1
  %35 = getelementptr inbounds i8, ptr addrspace(3) %26, i64 192
  store double %.unpack15131, ptr addrspace(3) %35, align 8
  %.repack18 = getelementptr inbounds i8, ptr addrspace(3) %26, i64 200
  store double %.unpack17132, ptr addrspace(3) %.repack18, align 8
  %36 = getelementptr inbounds i8, ptr addrspace(1) %21, i64 16384
  %37 = load <2 x double>, ptr addrspace(1) %36, align 16, !invariant.load !4
  %.unpack20133 = extractelement <2 x double> %37, i32 0
  %.unpack22134 = extractelement <2 x double> %37, i32 1
  %38 = getelementptr inbounds i8, ptr addrspace(3) %26, i64 256
  store double %.unpack20133, ptr addrspace(3) %38, align 8
  %.repack23 = getelementptr inbounds i8, ptr addrspace(3) %26, i64 264
  store double %.unpack22134, ptr addrspace(3) %.repack23, align 8
  %39 = getelementptr inbounds i8, ptr addrspace(1) %21, i64 20480
  %40 = load <2 x double>, ptr addrspace(1) %39, align 16, !invariant.load !4
  %.unpack25135 = extractelement <2 x double> %40, i32 0
  %.unpack27136 = extractelement <2 x double> %40, i32 1
  %41 = getelementptr inbounds i8, ptr addrspace(3) %26, i64 320
  store double %.unpack25135, ptr addrspace(3) %41, align 8
  %.repack28 = getelementptr inbounds i8, ptr addrspace(3) %26, i64 328
  store double %.unpack27136, ptr addrspace(3) %.repack28, align 8
  %42 = getelementptr inbounds i8, ptr addrspace(1) %21, i64 24576
  %43 = load <2 x double>, ptr addrspace(1) %42, align 16, !invariant.load !4
  %.unpack30137 = extractelement <2 x double> %43, i32 0
  %.unpack32138 = extractelement <2 x double> %43, i32 1
  %44 = getelementptr inbounds i8, ptr addrspace(3) %26, i64 384
  store double %.unpack30137, ptr addrspace(3) %44, align 8
  %.repack33 = getelementptr inbounds i8, ptr addrspace(3) %26, i64 392
  store double %.unpack32138, ptr addrspace(3) %.repack33, align 8
  %45 = getelementptr inbounds i8, ptr addrspace(1) %21, i64 28672
  %46 = load <2 x double>, ptr addrspace(1) %45, align 16, !invariant.load !4
  %.unpack35139 = extractelement <2 x double> %46, i32 0
  %.unpack37140 = extractelement <2 x double> %46, i32 1
  %47 = getelementptr inbounds i8, ptr addrspace(3) %26, i64 448
  store double %.unpack35139, ptr addrspace(3) %47, align 8
  %.repack38 = getelementptr inbounds i8, ptr addrspace(3) %26, i64 456
  store double %.unpack37140, ptr addrspace(3) %.repack38, align 8
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %48 = load i8, ptr addrspace(1) %6, align 256, !invariant.load !4
  %49 = trunc i8 %48 to i1
  %50 = mul nuw nsw i32 %12, 33
  %51 = add nuw nsw i32 %50, %18
  %52 = zext nneg i32 %51 to i64
  %53 = getelementptr inbounds [16 x i8], ptr addrspace(3) @shared_0, i64 %52
  %.unpack40 = load double, ptr addrspace(3) %53, align 8
  %.elt41 = getelementptr inbounds i8, ptr addrspace(3) %53, i64 8
  %.unpack42 = load double, ptr addrspace(3) %.elt41, align 8
  %54 = shl nuw nsw i32 %10, 11
  %55 = shl nuw nsw i32 %14, 5
  %56 = or disjoint i32 %54, %55
  %57 = or disjoint i32 %56, %13
  %58 = or disjoint i32 %57, %18
  %59 = zext nneg i32 %58 to i64
  %60 = getelementptr inbounds [16 x i8], ptr addrspace(1) %7, i64 %59
  %61 = load <2 x double>, ptr addrspace(1) %60, align 16
  %.unpack43141 = extractelement <2 x double> %61, i32 0
  %.unpack45142 = extractelement <2 x double> %61, i32 1
  %.elt = select i1 %49, double %.unpack40, double %.unpack43141
  %.elt47 = select i1 %49, double %.unpack42, double %.unpack45142
  %62 = insertelement <2 x double> poison, double %.elt, i32 0
  %63 = insertelement <2 x double> %62, double %.elt47, i32 1
  store <2 x double> %63, ptr addrspace(1) %60, align 16
  %64 = getelementptr inbounds i8, ptr addrspace(3) %53, i64 2112
  %.unpack49 = load double, ptr addrspace(3) %64, align 8
  %.elt50 = getelementptr inbounds i8, ptr addrspace(3) %53, i64 2120
  %.unpack51 = load double, ptr addrspace(3) %.elt50, align 8
  %65 = getelementptr inbounds i8, ptr addrspace(1) %60, i64 4096
  %66 = load <2 x double>, ptr addrspace(1) %65, align 16
  %.unpack53143 = extractelement <2 x double> %66, i32 0
  %.unpack55144 = extractelement <2 x double> %66, i32 1
  %.elt56 = select i1 %49, double %.unpack49, double %.unpack53143
  %.elt58 = select i1 %49, double %.unpack51, double %.unpack55144
  %67 = insertelement <2 x double> poison, double %.elt56, i32 0
  %68 = insertelement <2 x double> %67, double %.elt58, i32 1
  store <2 x double> %68, ptr addrspace(1) %65, align 16
  %69 = getelementptr inbounds i8, ptr addrspace(3) %53, i64 4224
  %.unpack60 = load double, ptr addrspace(3) %69, align 8
  %.elt61 = getelementptr inbounds i8, ptr addrspace(3) %53, i64 4232
  %.unpack62 = load double, ptr addrspace(3) %.elt61, align 8
  %70 = getelementptr inbounds i8, ptr addrspace(1) %60, i64 8192
  %71 = load <2 x double>, ptr addrspace(1) %70, align 16
  %.unpack64145 = extractelement <2 x double> %71, i32 0
  %.unpack66146 = extractelement <2 x double> %71, i32 1
  %.elt67 = select i1 %49, double %.unpack60, double %.unpack64145
  %.elt69 = select i1 %49, double %.unpack62, double %.unpack66146
  %72 = insertelement <2 x double> poison, double %.elt67, i32 0
  %73 = insertelement <2 x double> %72, double %.elt69, i32 1
  store <2 x double> %73, ptr addrspace(1) %70, align 16
  %74 = getelementptr inbounds i8, ptr addrspace(3) %53, i64 6336
  %.unpack71 = load double, ptr addrspace(3) %74, align 8
  %.elt72 = getelementptr inbounds i8, ptr addrspace(3) %53, i64 6344
  %.unpack73 = load double, ptr addrspace(3) %.elt72, align 8
  %75 = getelementptr inbounds i8, ptr addrspace(1) %60, i64 12288
  %76 = load <2 x double>, ptr addrspace(1) %75, align 16
  %.unpack75147 = extractelement <2 x double> %76, i32 0
  %.unpack77148 = extractelement <2 x double> %76, i32 1
  %.elt78 = select i1 %49, double %.unpack71, double %.unpack75147
  %.elt80 = select i1 %49, double %.unpack73, double %.unpack77148
  %77 = insertelement <2 x double> poison, double %.elt78, i32 0
  %78 = insertelement <2 x double> %77, double %.elt80, i32 1
  store <2 x double> %78, ptr addrspace(1) %75, align 16
  %79 = getelementptr inbounds i8, ptr addrspace(3) %53, i64 8448
  %.unpack82 = load double, ptr addrspace(3) %79, align 8
  %.elt83 = getelementptr inbounds i8, ptr addrspace(3) %53, i64 8456
  %.unpack84 = load double, ptr addrspace(3) %.elt83, align 8
  %80 = getelementptr inbounds i8, ptr addrspace(1) %60, i64 16384
  %81 = load <2 x double>, ptr addrspace(1) %80, align 16
  %.unpack86149 = extractelement <2 x double> %81, i32 0
  %.unpack88150 = extractelement <2 x double> %81, i32 1
  %.elt89 = select i1 %49, double %.unpack82, double %.unpack86149
  %.elt91 = select i1 %49, double %.unpack84, double %.unpack88150
  %82 = insertelement <2 x double> poison, double %.elt89, i32 0
  %83 = insertelement <2 x double> %82, double %.elt91, i32 1
  store <2 x double> %83, ptr addrspace(1) %80, align 16
  %84 = getelementptr inbounds i8, ptr addrspace(3) %53, i64 10560
  %.unpack93 = load double, ptr addrspace(3) %84, align 8
  %.elt94 = getelementptr inbounds i8, ptr addrspace(3) %53, i64 10568
  %.unpack95 = load double, ptr addrspace(3) %.elt94, align 8
  %85 = getelementptr inbounds i8, ptr addrspace(1) %60, i64 20480
  %86 = load <2 x double>, ptr addrspace(1) %85, align 16
  %.unpack97151 = extractelement <2 x double> %86, i32 0
  %.unpack99152 = extractelement <2 x double> %86, i32 1
  %.elt100 = select i1 %49, double %.unpack93, double %.unpack97151
  %.elt102 = select i1 %49, double %.unpack95, double %.unpack99152
  %87 = insertelement <2 x double> poison, double %.elt100, i32 0
  %88 = insertelement <2 x double> %87, double %.elt102, i32 1
  store <2 x double> %88, ptr addrspace(1) %85, align 16
  %89 = getelementptr inbounds i8, ptr addrspace(3) %53, i64 12672
  %.unpack104 = load double, ptr addrspace(3) %89, align 8
  %.elt105 = getelementptr inbounds i8, ptr addrspace(3) %53, i64 12680
  %.unpack106 = load double, ptr addrspace(3) %.elt105, align 8
  %90 = getelementptr inbounds i8, ptr addrspace(1) %60, i64 24576
  %91 = load <2 x double>, ptr addrspace(1) %90, align 16
  %.unpack108153 = extractelement <2 x double> %91, i32 0
  %.unpack110154 = extractelement <2 x double> %91, i32 1
  %.elt111 = select i1 %49, double %.unpack104, double %.unpack108153
  %.elt113 = select i1 %49, double %.unpack106, double %.unpack110154
  %92 = insertelement <2 x double> poison, double %.elt111, i32 0
  %93 = insertelement <2 x double> %92, double %.elt113, i32 1
  store <2 x double> %93, ptr addrspace(1) %90, align 16
  %94 = getelementptr inbounds i8, ptr addrspace(3) %53, i64 14784
  %.unpack115 = load double, ptr addrspace(3) %94, align 8
  %.elt116 = getelementptr inbounds i8, ptr addrspace(3) %53, i64 14792
  %.unpack117 = load double, ptr addrspace(3) %.elt116, align 8
  %95 = getelementptr inbounds i8, ptr addrspace(1) %60, i64 28672
  %96 = load <2 x double>, ptr addrspace(1) %95, align 16
  %.unpack119155 = extractelement <2 x double> %96, i32 0
  %.unpack121156 = extractelement <2 x double> %96, i32 1
  %.elt122 = select i1 %49, double %.unpack115, double %.unpack119155
  %.elt124 = select i1 %49, double %.unpack117, double %.unpack121156
  %97 = insertelement <2 x double> poison, double %.elt122, i32 0
  %98 = insertelement <2 x double> %97, double %.elt124, i32 1
  store <2 x double> %98, ptr addrspace(1) %95, align 16
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: convergent nocallback nounwind
declare void @llvm.nvvm.barrier.cta.sync.aligned.all(i32) #2

attributes #0 = { norecurse nounwind "nvvm.reqntid"="128,1,1" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent nocallback nounwind }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{i32 0, i32 128}
!3 = !{i32 0, i32 4}
!4 = !{}
