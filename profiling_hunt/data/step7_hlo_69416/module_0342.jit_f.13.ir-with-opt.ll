; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@shared_0 = private unnamed_addr addrspace(3) global [1056 x { double, double }] undef

; Function Attrs: norecurse nounwind
define ptx_kernel void @wrapped_transpose(ptr noalias nofree readonly align 256 captures(none) dereferenceable(65536) %0, ptr noalias nofree writeonly align 256 captures(none) dereferenceable(65536) %1) local_unnamed_addr #0 {
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
  %.unpack80 = extractelement <2 x double> %19, i32 0
  %.unpack281 = extractelement <2 x double> %19, i32 1
  %20 = mul nuw nsw i32 %15, 33
  %21 = add nuw nsw i32 %20, %9
  %22 = zext nneg i32 %21 to i64
  %23 = getelementptr inbounds [16 x i8], ptr addrspace(3) @shared_0, i64 %22
  store double %.unpack80, ptr addrspace(3) %23, align 8
  %.repack3 = getelementptr inbounds i8, ptr addrspace(3) %23, i64 8
  store double %.unpack281, ptr addrspace(3) %.repack3, align 8
  %24 = getelementptr inbounds i8, ptr addrspace(1) %18, i64 4096
  %25 = load <2 x double>, ptr addrspace(1) %24, align 16, !invariant.load !4
  %.unpack582 = extractelement <2 x double> %25, i32 0
  %.unpack783 = extractelement <2 x double> %25, i32 1
  %26 = getelementptr inbounds i8, ptr addrspace(3) %23, i64 64
  store double %.unpack582, ptr addrspace(3) %26, align 8
  %.repack8 = getelementptr inbounds i8, ptr addrspace(3) %23, i64 72
  store double %.unpack783, ptr addrspace(3) %.repack8, align 8
  %27 = getelementptr inbounds i8, ptr addrspace(1) %18, i64 8192
  %28 = load <2 x double>, ptr addrspace(1) %27, align 16, !invariant.load !4
  %.unpack1084 = extractelement <2 x double> %28, i32 0
  %.unpack1285 = extractelement <2 x double> %28, i32 1
  %29 = getelementptr inbounds i8, ptr addrspace(3) %23, i64 128
  store double %.unpack1084, ptr addrspace(3) %29, align 8
  %.repack13 = getelementptr inbounds i8, ptr addrspace(3) %23, i64 136
  store double %.unpack1285, ptr addrspace(3) %.repack13, align 8
  %30 = getelementptr inbounds i8, ptr addrspace(1) %18, i64 12288
  %31 = load <2 x double>, ptr addrspace(1) %30, align 16, !invariant.load !4
  %.unpack1586 = extractelement <2 x double> %31, i32 0
  %.unpack1787 = extractelement <2 x double> %31, i32 1
  %32 = getelementptr inbounds i8, ptr addrspace(3) %23, i64 192
  store double %.unpack1586, ptr addrspace(3) %32, align 8
  %.repack18 = getelementptr inbounds i8, ptr addrspace(3) %23, i64 200
  store double %.unpack1787, ptr addrspace(3) %.repack18, align 8
  %33 = getelementptr inbounds i8, ptr addrspace(1) %18, i64 16384
  %34 = load <2 x double>, ptr addrspace(1) %33, align 16, !invariant.load !4
  %.unpack2088 = extractelement <2 x double> %34, i32 0
  %.unpack2289 = extractelement <2 x double> %34, i32 1
  %35 = getelementptr inbounds i8, ptr addrspace(3) %23, i64 256
  store double %.unpack2088, ptr addrspace(3) %35, align 8
  %.repack23 = getelementptr inbounds i8, ptr addrspace(3) %23, i64 264
  store double %.unpack2289, ptr addrspace(3) %.repack23, align 8
  %36 = getelementptr inbounds i8, ptr addrspace(1) %18, i64 20480
  %37 = load <2 x double>, ptr addrspace(1) %36, align 16, !invariant.load !4
  %.unpack2590 = extractelement <2 x double> %37, i32 0
  %.unpack2791 = extractelement <2 x double> %37, i32 1
  %38 = getelementptr inbounds i8, ptr addrspace(3) %23, i64 320
  store double %.unpack2590, ptr addrspace(3) %38, align 8
  %.repack28 = getelementptr inbounds i8, ptr addrspace(3) %23, i64 328
  store double %.unpack2791, ptr addrspace(3) %.repack28, align 8
  %39 = getelementptr inbounds i8, ptr addrspace(1) %18, i64 24576
  %40 = load <2 x double>, ptr addrspace(1) %39, align 16, !invariant.load !4
  %.unpack3092 = extractelement <2 x double> %40, i32 0
  %.unpack3293 = extractelement <2 x double> %40, i32 1
  %41 = getelementptr inbounds i8, ptr addrspace(3) %23, i64 384
  store double %.unpack3092, ptr addrspace(3) %41, align 8
  %.repack33 = getelementptr inbounds i8, ptr addrspace(3) %23, i64 392
  store double %.unpack3293, ptr addrspace(3) %.repack33, align 8
  %42 = getelementptr inbounds i8, ptr addrspace(1) %18, i64 28672
  %43 = load <2 x double>, ptr addrspace(1) %42, align 16, !invariant.load !4
  %.unpack3594 = extractelement <2 x double> %43, i32 0
  %.unpack3795 = extractelement <2 x double> %43, i32 1
  %44 = getelementptr inbounds i8, ptr addrspace(3) %23, i64 448
  store double %.unpack3594, ptr addrspace(3) %44, align 8
  %.repack38 = getelementptr inbounds i8, ptr addrspace(3) %23, i64 456
  store double %.unpack3795, ptr addrspace(3) %.repack38, align 8
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
  %55 = getelementptr inbounds [16 x i8], ptr addrspace(1) %4, i64 %54
  %56 = insertelement <2 x double> poison, double %.unpack40, i32 0
  %57 = insertelement <2 x double> %56, double %.unpack42, i32 1
  store <2 x double> %57, ptr addrspace(1) %55, align 16
  %58 = getelementptr inbounds i8, ptr addrspace(3) %48, i64 2112
  %.unpack45 = load double, ptr addrspace(3) %58, align 8
  %.elt46 = getelementptr inbounds i8, ptr addrspace(3) %48, i64 2120
  %.unpack47 = load double, ptr addrspace(3) %.elt46, align 8
  %59 = getelementptr inbounds i8, ptr addrspace(1) %55, i64 4096
  %60 = insertelement <2 x double> poison, double %.unpack45, i32 0
  %61 = insertelement <2 x double> %60, double %.unpack47, i32 1
  store <2 x double> %61, ptr addrspace(1) %59, align 16
  %62 = getelementptr inbounds i8, ptr addrspace(3) %48, i64 4224
  %.unpack50 = load double, ptr addrspace(3) %62, align 8
  %.elt51 = getelementptr inbounds i8, ptr addrspace(3) %48, i64 4232
  %.unpack52 = load double, ptr addrspace(3) %.elt51, align 8
  %63 = getelementptr inbounds i8, ptr addrspace(1) %55, i64 8192
  %64 = insertelement <2 x double> poison, double %.unpack50, i32 0
  %65 = insertelement <2 x double> %64, double %.unpack52, i32 1
  store <2 x double> %65, ptr addrspace(1) %63, align 16
  %66 = getelementptr inbounds i8, ptr addrspace(3) %48, i64 6336
  %.unpack55 = load double, ptr addrspace(3) %66, align 8
  %.elt56 = getelementptr inbounds i8, ptr addrspace(3) %48, i64 6344
  %.unpack57 = load double, ptr addrspace(3) %.elt56, align 8
  %67 = getelementptr inbounds i8, ptr addrspace(1) %55, i64 12288
  %68 = insertelement <2 x double> poison, double %.unpack55, i32 0
  %69 = insertelement <2 x double> %68, double %.unpack57, i32 1
  store <2 x double> %69, ptr addrspace(1) %67, align 16
  %70 = getelementptr inbounds i8, ptr addrspace(3) %48, i64 8448
  %.unpack60 = load double, ptr addrspace(3) %70, align 8
  %.elt61 = getelementptr inbounds i8, ptr addrspace(3) %48, i64 8456
  %.unpack62 = load double, ptr addrspace(3) %.elt61, align 8
  %71 = getelementptr inbounds i8, ptr addrspace(1) %55, i64 16384
  %72 = insertelement <2 x double> poison, double %.unpack60, i32 0
  %73 = insertelement <2 x double> %72, double %.unpack62, i32 1
  store <2 x double> %73, ptr addrspace(1) %71, align 16
  %74 = getelementptr inbounds i8, ptr addrspace(3) %48, i64 10560
  %.unpack65 = load double, ptr addrspace(3) %74, align 8
  %.elt66 = getelementptr inbounds i8, ptr addrspace(3) %48, i64 10568
  %.unpack67 = load double, ptr addrspace(3) %.elt66, align 8
  %75 = getelementptr inbounds i8, ptr addrspace(1) %55, i64 20480
  %76 = insertelement <2 x double> poison, double %.unpack65, i32 0
  %77 = insertelement <2 x double> %76, double %.unpack67, i32 1
  store <2 x double> %77, ptr addrspace(1) %75, align 16
  %78 = getelementptr inbounds i8, ptr addrspace(3) %48, i64 12672
  %.unpack70 = load double, ptr addrspace(3) %78, align 8
  %.elt71 = getelementptr inbounds i8, ptr addrspace(3) %48, i64 12680
  %.unpack72 = load double, ptr addrspace(3) %.elt71, align 8
  %79 = getelementptr inbounds i8, ptr addrspace(1) %55, i64 24576
  %80 = insertelement <2 x double> poison, double %.unpack70, i32 0
  %81 = insertelement <2 x double> %80, double %.unpack72, i32 1
  store <2 x double> %81, ptr addrspace(1) %79, align 16
  %82 = getelementptr inbounds i8, ptr addrspace(3) %48, i64 14784
  %.unpack75 = load double, ptr addrspace(3) %82, align 8
  %.elt76 = getelementptr inbounds i8, ptr addrspace(3) %48, i64 14792
  %.unpack77 = load double, ptr addrspace(3) %.elt76, align 8
  %83 = getelementptr inbounds i8, ptr addrspace(1) %55, i64 28672
  %84 = insertelement <2 x double> poison, double %.unpack75, i32 0
  %85 = insertelement <2 x double> %84, double %.unpack77, i32 1
  store <2 x double> %85, ptr addrspace(1) %83, align 16
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
