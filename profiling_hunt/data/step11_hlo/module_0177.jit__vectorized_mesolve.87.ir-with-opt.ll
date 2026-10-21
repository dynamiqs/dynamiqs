; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@shared_0 = private unnamed_addr addrspace(3) global [1056 x { double, double }] undef

; Function Attrs: norecurse nounwind
define ptx_kernel void @input_transpose_fusion(ptr noalias nofree readonly align 256 captures(none) dereferenceable(65536) %0, ptr noalias nofree writeonly align 256 captures(none) dereferenceable(65536) %1) local_unnamed_addr #0 {
  %3 = addrspacecast ptr %0 to ptr addrspace(1)
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %6 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !3
  %7 = and i32 %6, 1
  %8 = shl nuw nsw i32 %7, 5
  %9 = lshr i32 %5, 5
  %10 = shl nuw nsw i32 %9, 6
  %11 = lshr i32 %6, 1
  %12 = shl nuw nsw i32 %11, 11
  %13 = or disjoint i32 %8, %12
  %14 = or disjoint i32 %13, %10
  %15 = and i32 %5, 31
  %16 = or disjoint i32 %14, %15
  %17 = zext nneg i32 %16 to i64
  %18 = getelementptr inbounds [16 x i8], ptr addrspace(1) %3, i64 %17
  %19 = load <2 x double>, ptr addrspace(1) %18, align 16, !invariant.load !4
  %.unpack118 = extractelement <2 x double> %19, i32 0
  %.unpack2119 = extractelement <2 x double> %19, i32 1
  %20 = mul nuw nsw i32 %15, 33
  %21 = add nuw nsw i32 %20, %9
  %22 = zext nneg i32 %21 to i64
  %23 = getelementptr inbounds [16 x i8], ptr addrspace(3) @shared_0, i64 %22
  store double %.unpack118, ptr addrspace(3) %23, align 8
  %.repack3 = getelementptr inbounds i8, ptr addrspace(3) %23, i64 8
  store double %.unpack2119, ptr addrspace(3) %.repack3, align 8
  %24 = getelementptr inbounds i8, ptr addrspace(1) %18, i64 4096
  %25 = load <2 x double>, ptr addrspace(1) %24, align 16, !invariant.load !4
  %.unpack5120 = extractelement <2 x double> %25, i32 0
  %.unpack7121 = extractelement <2 x double> %25, i32 1
  %26 = getelementptr inbounds i8, ptr addrspace(3) %23, i64 64
  store double %.unpack5120, ptr addrspace(3) %26, align 8
  %.repack8 = getelementptr inbounds i8, ptr addrspace(3) %23, i64 72
  store double %.unpack7121, ptr addrspace(3) %.repack8, align 8
  %27 = getelementptr inbounds i8, ptr addrspace(1) %18, i64 8192
  %28 = load <2 x double>, ptr addrspace(1) %27, align 16, !invariant.load !4
  %.unpack10122 = extractelement <2 x double> %28, i32 0
  %.unpack12123 = extractelement <2 x double> %28, i32 1
  %29 = getelementptr inbounds i8, ptr addrspace(3) %23, i64 128
  store double %.unpack10122, ptr addrspace(3) %29, align 8
  %.repack13 = getelementptr inbounds i8, ptr addrspace(3) %23, i64 136
  store double %.unpack12123, ptr addrspace(3) %.repack13, align 8
  %30 = getelementptr inbounds i8, ptr addrspace(1) %18, i64 12288
  %31 = load <2 x double>, ptr addrspace(1) %30, align 16, !invariant.load !4
  %.unpack15124 = extractelement <2 x double> %31, i32 0
  %.unpack17125 = extractelement <2 x double> %31, i32 1
  %32 = getelementptr inbounds i8, ptr addrspace(3) %23, i64 192
  store double %.unpack15124, ptr addrspace(3) %32, align 8
  %.repack18 = getelementptr inbounds i8, ptr addrspace(3) %23, i64 200
  store double %.unpack17125, ptr addrspace(3) %.repack18, align 8
  %33 = getelementptr inbounds i8, ptr addrspace(1) %18, i64 16384
  %34 = load <2 x double>, ptr addrspace(1) %33, align 16, !invariant.load !4
  %.unpack20126 = extractelement <2 x double> %34, i32 0
  %.unpack22127 = extractelement <2 x double> %34, i32 1
  %35 = getelementptr inbounds i8, ptr addrspace(3) %23, i64 256
  store double %.unpack20126, ptr addrspace(3) %35, align 8
  %.repack23 = getelementptr inbounds i8, ptr addrspace(3) %23, i64 264
  store double %.unpack22127, ptr addrspace(3) %.repack23, align 8
  %36 = getelementptr inbounds i8, ptr addrspace(1) %18, i64 20480
  %37 = load <2 x double>, ptr addrspace(1) %36, align 16, !invariant.load !4
  %.unpack25128 = extractelement <2 x double> %37, i32 0
  %.unpack27129 = extractelement <2 x double> %37, i32 1
  %38 = getelementptr inbounds i8, ptr addrspace(3) %23, i64 320
  store double %.unpack25128, ptr addrspace(3) %38, align 8
  %.repack28 = getelementptr inbounds i8, ptr addrspace(3) %23, i64 328
  store double %.unpack27129, ptr addrspace(3) %.repack28, align 8
  %39 = getelementptr inbounds i8, ptr addrspace(1) %18, i64 24576
  %40 = load <2 x double>, ptr addrspace(1) %39, align 16, !invariant.load !4
  %.unpack30130 = extractelement <2 x double> %40, i32 0
  %.unpack32131 = extractelement <2 x double> %40, i32 1
  %41 = getelementptr inbounds i8, ptr addrspace(3) %23, i64 384
  store double %.unpack30130, ptr addrspace(3) %41, align 8
  %.repack33 = getelementptr inbounds i8, ptr addrspace(3) %23, i64 392
  store double %.unpack32131, ptr addrspace(3) %.repack33, align 8
  %42 = getelementptr inbounds i8, ptr addrspace(1) %18, i64 28672
  %43 = load <2 x double>, ptr addrspace(1) %42, align 16, !invariant.load !4
  %.unpack35132 = extractelement <2 x double> %43, i32 0
  %.unpack37133 = extractelement <2 x double> %43, i32 1
  %44 = getelementptr inbounds i8, ptr addrspace(3) %23, i64 448
  store double %.unpack35132, ptr addrspace(3) %44, align 8
  %.repack38 = getelementptr inbounds i8, ptr addrspace(3) %23, i64 456
  store double %.unpack37133, ptr addrspace(3) %.repack38, align 8
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %45 = mul nuw nsw i32 %9, 33
  %46 = add nuw nsw i32 %45, %15
  %47 = zext nneg i32 %46 to i64
  %48 = getelementptr inbounds [16 x i8], ptr addrspace(3) @shared_0, i64 %47
  %.unpack40 = load double, ptr addrspace(3) %48, align 8
  %.elt41 = getelementptr inbounds i8, ptr addrspace(3) %48, i64 8
  %.unpack42 = load double, ptr addrspace(3) %.elt41, align 8
  %49 = shl nuw nsw i32 %7, 11
  %50 = shl nuw nsw i32 %11, 5
  %51 = or disjoint i32 %49, %50
  %52 = or disjoint i32 %51, %10
  %53 = or disjoint i32 %52, %15
  %54 = zext nneg i32 %53 to i64
  %55 = getelementptr inbounds [16 x i8], ptr addrspace(1) %3, i64 %54
  %56 = load <2 x double>, ptr addrspace(1) %55, align 16, !invariant.load !4
  %.unpack43134 = extractelement <2 x double> %56, i32 0
  %.unpack45135 = extractelement <2 x double> %56, i32 1
  %57 = fadd double %.unpack40, %.unpack43134
  %58 = fsub double %.unpack42, %.unpack45135
  %59 = getelementptr inbounds [16 x i8], ptr addrspace(1) %4, i64 %54
  %60 = insertelement <2 x double> poison, double %57, i32 0
  %61 = insertelement <2 x double> %60, double %58, i32 1
  store <2 x double> %61, ptr addrspace(1) %59, align 16
  %62 = getelementptr inbounds i8, ptr addrspace(3) %48, i64 2112
  %.unpack48 = load double, ptr addrspace(3) %62, align 8
  %.elt49 = getelementptr inbounds i8, ptr addrspace(3) %48, i64 2120
  %.unpack50 = load double, ptr addrspace(3) %.elt49, align 8
  %63 = getelementptr inbounds i8, ptr addrspace(1) %55, i64 4096
  %64 = load <2 x double>, ptr addrspace(1) %63, align 16, !invariant.load !4
  %.unpack51136 = extractelement <2 x double> %64, i32 0
  %.unpack53137 = extractelement <2 x double> %64, i32 1
  %65 = fadd double %.unpack48, %.unpack51136
  %66 = fsub double %.unpack50, %.unpack53137
  %67 = getelementptr inbounds i8, ptr addrspace(1) %59, i64 4096
  %68 = insertelement <2 x double> poison, double %65, i32 0
  %69 = insertelement <2 x double> %68, double %66, i32 1
  store <2 x double> %69, ptr addrspace(1) %67, align 16
  %70 = getelementptr inbounds i8, ptr addrspace(3) %48, i64 4224
  %.unpack56 = load double, ptr addrspace(3) %70, align 8
  %.elt57 = getelementptr inbounds i8, ptr addrspace(3) %48, i64 4232
  %.unpack58 = load double, ptr addrspace(3) %.elt57, align 8
  %71 = getelementptr inbounds i8, ptr addrspace(1) %55, i64 8192
  %72 = load <2 x double>, ptr addrspace(1) %71, align 16, !invariant.load !4
  %.unpack59138 = extractelement <2 x double> %72, i32 0
  %.unpack61139 = extractelement <2 x double> %72, i32 1
  %73 = fadd double %.unpack56, %.unpack59138
  %74 = fsub double %.unpack58, %.unpack61139
  %75 = getelementptr inbounds i8, ptr addrspace(1) %59, i64 8192
  %76 = insertelement <2 x double> poison, double %73, i32 0
  %77 = insertelement <2 x double> %76, double %74, i32 1
  store <2 x double> %77, ptr addrspace(1) %75, align 16
  %78 = getelementptr inbounds i8, ptr addrspace(3) %48, i64 6336
  %.unpack64 = load double, ptr addrspace(3) %78, align 8
  %.elt65 = getelementptr inbounds i8, ptr addrspace(3) %48, i64 6344
  %.unpack66 = load double, ptr addrspace(3) %.elt65, align 8
  %79 = getelementptr inbounds i8, ptr addrspace(1) %55, i64 12288
  %80 = load <2 x double>, ptr addrspace(1) %79, align 16, !invariant.load !4
  %.unpack67140 = extractelement <2 x double> %80, i32 0
  %.unpack69141 = extractelement <2 x double> %80, i32 1
  %81 = fadd double %.unpack64, %.unpack67140
  %82 = fsub double %.unpack66, %.unpack69141
  %83 = getelementptr inbounds i8, ptr addrspace(1) %59, i64 12288
  %84 = insertelement <2 x double> poison, double %81, i32 0
  %85 = insertelement <2 x double> %84, double %82, i32 1
  store <2 x double> %85, ptr addrspace(1) %83, align 16
  %86 = getelementptr inbounds i8, ptr addrspace(3) %48, i64 8448
  %.unpack72 = load double, ptr addrspace(3) %86, align 8
  %.elt73 = getelementptr inbounds i8, ptr addrspace(3) %48, i64 8456
  %.unpack74 = load double, ptr addrspace(3) %.elt73, align 8
  %87 = getelementptr inbounds i8, ptr addrspace(1) %55, i64 16384
  %88 = load <2 x double>, ptr addrspace(1) %87, align 16, !invariant.load !4
  %.unpack75142 = extractelement <2 x double> %88, i32 0
  %.unpack77143 = extractelement <2 x double> %88, i32 1
  %89 = fadd double %.unpack72, %.unpack75142
  %90 = fsub double %.unpack74, %.unpack77143
  %91 = getelementptr inbounds i8, ptr addrspace(1) %59, i64 16384
  %92 = insertelement <2 x double> poison, double %89, i32 0
  %93 = insertelement <2 x double> %92, double %90, i32 1
  store <2 x double> %93, ptr addrspace(1) %91, align 16
  %94 = getelementptr inbounds i8, ptr addrspace(3) %48, i64 10560
  %.unpack80 = load double, ptr addrspace(3) %94, align 8
  %.elt81 = getelementptr inbounds i8, ptr addrspace(3) %48, i64 10568
  %.unpack82 = load double, ptr addrspace(3) %.elt81, align 8
  %95 = getelementptr inbounds i8, ptr addrspace(1) %55, i64 20480
  %96 = load <2 x double>, ptr addrspace(1) %95, align 16, !invariant.load !4
  %.unpack83144 = extractelement <2 x double> %96, i32 0
  %.unpack85145 = extractelement <2 x double> %96, i32 1
  %97 = fadd double %.unpack80, %.unpack83144
  %98 = fsub double %.unpack82, %.unpack85145
  %99 = getelementptr inbounds i8, ptr addrspace(1) %59, i64 20480
  %100 = insertelement <2 x double> poison, double %97, i32 0
  %101 = insertelement <2 x double> %100, double %98, i32 1
  store <2 x double> %101, ptr addrspace(1) %99, align 16
  %102 = getelementptr inbounds i8, ptr addrspace(3) %48, i64 12672
  %.unpack88 = load double, ptr addrspace(3) %102, align 8
  %.elt89 = getelementptr inbounds i8, ptr addrspace(3) %48, i64 12680
  %.unpack90 = load double, ptr addrspace(3) %.elt89, align 8
  %103 = getelementptr inbounds i8, ptr addrspace(1) %55, i64 24576
  %104 = load <2 x double>, ptr addrspace(1) %103, align 16, !invariant.load !4
  %.unpack91146 = extractelement <2 x double> %104, i32 0
  %.unpack93147 = extractelement <2 x double> %104, i32 1
  %105 = fadd double %.unpack88, %.unpack91146
  %106 = fsub double %.unpack90, %.unpack93147
  %107 = getelementptr inbounds i8, ptr addrspace(1) %59, i64 24576
  %108 = insertelement <2 x double> poison, double %105, i32 0
  %109 = insertelement <2 x double> %108, double %106, i32 1
  store <2 x double> %109, ptr addrspace(1) %107, align 16
  %110 = getelementptr inbounds i8, ptr addrspace(3) %48, i64 14784
  %.unpack96 = load double, ptr addrspace(3) %110, align 8
  %.elt97 = getelementptr inbounds i8, ptr addrspace(3) %48, i64 14792
  %.unpack98 = load double, ptr addrspace(3) %.elt97, align 8
  %111 = getelementptr inbounds i8, ptr addrspace(1) %55, i64 28672
  %112 = load <2 x double>, ptr addrspace(1) %111, align 16, !invariant.load !4
  %.unpack99148 = extractelement <2 x double> %112, i32 0
  %.unpack101149 = extractelement <2 x double> %112, i32 1
  %113 = fadd double %.unpack96, %.unpack99148
  %114 = fsub double %.unpack98, %.unpack101149
  %115 = getelementptr inbounds i8, ptr addrspace(1) %59, i64 28672
  %116 = insertelement <2 x double> poison, double %113, i32 0
  %117 = insertelement <2 x double> %116, double %114, i32 1
  store <2 x double> %117, ptr addrspace(1) %115, align 16
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
