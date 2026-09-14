; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_multiply_fusion(ptr noalias nofree readonly align 16 captures(none) dereferenceable(1024) %0, ptr noalias nofree readonly align 16 captures(none) dereferenceable(1024) %1, ptr noalias nofree writeonly align 256 captures(none) dereferenceable(65536) %2) local_unnamed_addr #0 {
  %4 = addrspacecast ptr %0 to ptr addrspace(1)
  %5 = addrspacecast ptr %1 to ptr addrspace(1)
  %6 = addrspacecast ptr %2 to ptr addrspace(1)
  %7 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !2
  %8 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !3
  %9 = shl nuw nsw i32 %7, 1
  %10 = lshr i32 %8, 6
  %11 = or disjoint i32 %9, %10
  %12 = and i32 %8, 63
  %13 = zext nneg i32 %11 to i64
  %14 = getelementptr inbounds [16 x i8], ptr addrspace(1) %4, i64 %13
  %15 = load <2 x double>, ptr addrspace(1) %14, align 16, !invariant.load !4
  %.unpack8 = extractelement <2 x double> %15, i32 0
  %.unpack29 = extractelement <2 x double> %15, i32 1
  %16 = zext nneg i32 %12 to i64
  %17 = getelementptr inbounds [16 x i8], ptr addrspace(1) %5, i64 %16
  %18 = load <2 x double>, ptr addrspace(1) %17, align 16, !invariant.load !4
  %.unpack310 = extractelement <2 x double> %18, i32 0
  %.unpack511 = extractelement <2 x double> %18, i32 1
  %19 = fmul double %.unpack8, %.unpack310
  %20 = fmul double %.unpack29, %.unpack511
  %21 = fmul double %.unpack29, %.unpack310
  %22 = fmul double %.unpack8, %.unpack511
  %23 = fsub double %19, %20
  %24 = fadd double %21, %22
  %25 = shl nuw nsw i32 %7, 7
  %26 = or disjoint i32 %25, %8
  %27 = zext nneg i32 %26 to i64
  %28 = getelementptr inbounds [16 x i8], ptr addrspace(1) %6, i64 %27
  %29 = insertelement <2 x double> poison, double %23, i32 0
  %30 = insertelement <2 x double> %29, double %24, i32 1
  store <2 x double> %30, ptr addrspace(1) %28, align 16
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "nvvm.reqntid"="128,1,1" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{i32 0, i32 32}
!3 = !{i32 0, i32 128}
!4 = !{}
