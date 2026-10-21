; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_complex_fusion(ptr noalias nofree readonly align 16 captures(none) dereferenceable(4) %0, ptr noalias nofree readonly align 16 captures(none) dereferenceable(1008) %1, ptr noalias nofree readonly align 16 captures(none) dereferenceable(1024) %2, ptr noalias nofree writeonly align 256 captures(none) dereferenceable(1024) %3) local_unnamed_addr #0 {
  %5 = addrspacecast ptr %0 to ptr addrspace(1)
  %6 = addrspacecast ptr %1 to ptr addrspace(1)
  %7 = addrspacecast ptr %2 to ptr addrspace(1)
  %8 = addrspacecast ptr %3 to ptr addrspace(1)
  %9 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %10 = load i32, ptr addrspace(1) %5, align 16, !invariant.load !3
  %11 = icmp sgt i32 %10, 0
  %12 = zext i1 %11 to i32
  %13 = icmp samesign uge i32 %9, %12
  %14 = icmp ne i32 %9, 63
  %15 = or i1 %14, %11
  %16 = and i1 %15, %13
  br i1 %16, label %17, label %30

17:                                               ; preds = %4
  %18 = sub nuw nsw i32 %9, %12
  %19 = zext nneg i32 %18 to i64
  %20 = getelementptr inbounds [16 x i8], ptr addrspace(1) %6, i64 %19
  %21 = load <2 x double>, ptr addrspace(1) %20, align 16, !invariant.load !3
  %.unpack9 = extractelement <2 x double> %21, i32 0
  %.unpack210 = extractelement <2 x double> %21, i32 1
  %22 = icmp ult i32 %10, 2
  %23 = zext nneg i32 %9 to i64
  %24 = getelementptr inbounds [16 x i8], ptr addrspace(1) %7, i64 %23
  %25 = load <2 x double>, ptr addrspace(1) %24, align 16, !invariant.load !3
  %.unpack.i11 = extractelement <2 x double> %25, i32 0
  %.unpack2.i12 = extractelement <2 x double> %25, i32 1
  %26 = fadd double %.unpack9, %.unpack.i11
  %27 = select i1 %22, double %26, double %.unpack.i11
  %28 = fadd double %.unpack210, %.unpack2.i12
  %29 = select i1 %22, double %28, double %.unpack2.i12
  br label %34

30:                                               ; preds = %4
  %31 = zext nneg i32 %9 to i64
  %32 = getelementptr inbounds [16 x i8], ptr addrspace(1) %7, i64 %31
  %33 = load <2 x double>, ptr addrspace(1) %32, align 16, !invariant.load !3
  %.unpack.i513 = extractelement <2 x double> %33, i32 0
  %.unpack2.i714 = extractelement <2 x double> %33, i32 1
  br label %34

34:                                               ; preds = %30, %17
  %.pre-phi = phi i64 [ %31, %30 ], [ %23, %17 ]
  %35 = phi double [ %.unpack.i513, %30 ], [ %27, %17 ]
  %36 = phi double [ %.unpack2.i714, %30 ], [ %29, %17 ]
  %37 = getelementptr inbounds [16 x i8], ptr addrspace(1) %8, i64 %.pre-phi
  %38 = insertelement <2 x double> poison, double %35, i32 0
  %39 = insertelement <2 x double> %38, double %36, i32 1
  store <2 x double> %39, ptr addrspace(1) %37, align 16
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "nvvm.reqntid"="64,1,1" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{i32 0, i32 64}
!3 = !{}
