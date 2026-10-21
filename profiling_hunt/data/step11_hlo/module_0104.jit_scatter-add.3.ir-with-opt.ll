; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_complex_fusion(ptr noalias nofree readonly align 16 captures(none) dereferenceable(4) %0, ptr noalias nofree readonly align 16 captures(none) dereferenceable(2048) %1, ptr noalias nofree readonly align 16 captures(none) dereferenceable(1024) %2, ptr noalias nofree readonly align 256 captures(none) dereferenceable(1) %3, ptr noalias nofree writeonly align 256 captures(none) dereferenceable(2048) %4) local_unnamed_addr #0 {
  %6 = addrspacecast ptr %0 to ptr addrspace(1)
  %7 = addrspacecast ptr %2 to ptr addrspace(1)
  %8 = addrspacecast ptr %3 to ptr addrspace(1)
  %9 = addrspacecast ptr %1 to ptr addrspace(1)
  %10 = addrspacecast ptr %4 to ptr addrspace(1)
  %11 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %12 = lshr i32 %11, 6
  %13 = load i32, ptr addrspace(1) %6, align 16, !invariant.load !3
  %14 = icmp sgt i32 %13, 0
  %15 = zext i1 %14 to i32
  %16 = select i1 %14, i32 2, i32 1
  %17 = icmp samesign uge i32 %12, %15
  %18 = icmp samesign ult i32 %12, %16
  %19 = and i1 %17, %18
  br i1 %19, label %20, label %34

20:                                               ; preds = %5
  %21 = and i32 %11, 63
  %22 = zext nneg i32 %21 to i64
  %23 = getelementptr inbounds [16 x i8], ptr addrspace(1) %7, i64 %22
  %24 = load <2 x double>, ptr addrspace(1) %23, align 16, !invariant.load !3
  %.unpack8 = extractelement <2 x double> %24, i32 0
  %.unpack29 = extractelement <2 x double> %24, i32 1
  %25 = load i8, ptr addrspace(1) %8, align 256, !invariant.load !3
  %26 = zext nneg i32 %11 to i64
  %27 = getelementptr inbounds [16 x i8], ptr addrspace(1) %9, i64 %26
  %28 = load <2 x double>, ptr addrspace(1) %27, align 16, !invariant.load !3
  %.unpack.i10 = extractelement <2 x double> %28, i32 0
  %.unpack2.i11 = extractelement <2 x double> %28, i32 1
  %29 = fadd double %.unpack8, %.unpack.i10
  %30 = trunc i8 %25 to i1
  %31 = select i1 %30, double %29, double %.unpack.i10
  %32 = fadd double %.unpack29, %.unpack2.i11
  %33 = select i1 %30, double %32, double %.unpack2.i11
  br label %38

34:                                               ; preds = %5
  %35 = zext nneg i32 %11 to i64
  %36 = getelementptr inbounds [16 x i8], ptr addrspace(1) %9, i64 %35
  %37 = load <2 x double>, ptr addrspace(1) %36, align 16, !invariant.load !3
  %.unpack.i512 = extractelement <2 x double> %37, i32 0
  %.unpack2.i713 = extractelement <2 x double> %37, i32 1
  br label %38

38:                                               ; preds = %34, %20
  %.pre-phi = phi i64 [ %35, %34 ], [ %26, %20 ]
  %39 = phi double [ %.unpack.i512, %34 ], [ %31, %20 ]
  %40 = phi double [ %.unpack2.i713, %34 ], [ %33, %20 ]
  %41 = getelementptr inbounds [16 x i8], ptr addrspace(1) %10, i64 %.pre-phi
  %42 = insertelement <2 x double> poison, double %39, i32 0
  %43 = insertelement <2 x double> %42, double %40, i32 1
  store <2 x double> %43, ptr addrspace(1) %41, align 16
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "nvvm.reqntid"="128,1,1" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{i32 0, i32 128}
!3 = !{}
