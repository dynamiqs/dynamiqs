; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_pad_fusion(ptr noalias nofree readonly align 256 captures(none) dereferenceable(65536) %0, ptr noalias nofree readonly align 256 captures(none) dereferenceable(8) %1, ptr noalias nofree writeonly align 256 captures(none) dereferenceable(458752) %2) local_unnamed_addr #0 {
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = addrspacecast ptr %0 to ptr addrspace(1)
  %6 = addrspacecast ptr %2 to ptr addrspace(1)
  %7 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !2
  %8 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !3
  %9 = icmp samesign ult i32 %7, 32
  br i1 %9, label %10, label %._crit_edge

._crit_edge:                                      ; preds = %3
  %.pre = shl nuw nsw i32 %7, 7
  %.pre5 = or disjoint i32 %.pre, %8
  %.pre7 = zext nneg i32 %.pre5 to i64
  br label %25

10:                                               ; preds = %3
  %11 = load double, ptr addrspace(1) %4, align 256, !invariant.load !4
  %12 = shl nuw nsw i32 %7, 7
  %13 = or disjoint i32 %12, %8
  %14 = zext nneg i32 %13 to i64
  %15 = getelementptr inbounds [16 x i8], ptr addrspace(1) %5, i64 %14
  %16 = load <2 x double>, ptr addrspace(1) %15, align 16, !invariant.load !4
  %.unpack9 = extractelement <2 x double> %16, i32 0
  %.unpack210 = extractelement <2 x double> %16, i32 1
  %17 = fmul double %11, %.unpack9
  %18 = fmul double %.unpack210, 0.000000e+00
  %19 = fmul double %.unpack9, 0.000000e+00
  %20 = fmul double %11, %.unpack210
  %21 = fsub double %17, %18
  %22 = fadd double %19, %20
  %23 = insertvalue { double, double } poison, double %21, 0
  %24 = insertvalue { double, double } %23, double %22, 1
  br label %25

25:                                               ; preds = %._crit_edge, %10
  %.pre-phi8 = phi i64 [ %.pre7, %._crit_edge ], [ %14, %10 ]
  %26 = phi { double, double } [ zeroinitializer, %._crit_edge ], [ %24, %10 ]
  %27 = getelementptr inbounds [16 x i8], ptr addrspace(1) %6, i64 %.pre-phi8
  %.elt = extractvalue { double, double } %26, 0
  %.elt4 = extractvalue { double, double } %26, 1
  %28 = insertelement <2 x double> poison, double %.elt, i32 0
  %29 = insertelement <2 x double> %28, double %.elt4, i32 1
  store <2 x double> %29, ptr addrspace(1) %27, align 16
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
!2 = !{i32 0, i32 224}
!3 = !{i32 0, i32 128}
!4 = !{}
