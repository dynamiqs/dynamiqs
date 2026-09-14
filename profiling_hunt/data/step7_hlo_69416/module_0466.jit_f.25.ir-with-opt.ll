; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_dynamic_update_slice_fusion_1(ptr noalias nofree align 256 captures(none) dereferenceable(458752) %0, ptr noalias nofree readonly align 256 captures(none) dereferenceable(65536) %1, ptr noalias nofree readonly align 256 captures(none) dereferenceable(8) %2, ptr noalias nofree readonly align 256 captures(none) dereferenceable(8) %3, ptr noalias nofree readonly align 256 captures(none) dereferenceable(8) %4, ptr noalias nofree readonly align 256 captures(none) dereferenceable(12) %5, ptr noalias nofree readnone align 256 captures(none) dereferenceable(458752) %6) local_unnamed_addr #0 {
  %8 = addrspacecast ptr %3 to ptr addrspace(1)
  %9 = addrspacecast ptr %4 to ptr addrspace(1)
  %10 = addrspacecast ptr %5 to ptr addrspace(1)
  %11 = addrspacecast ptr %2 to ptr addrspace(1)
  %12 = addrspacecast ptr %1 to ptr addrspace(1)
  %13 = addrspacecast ptr %0 to ptr addrspace(1)
  %14 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !2
  %15 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !3
  %16 = load i64, ptr addrspace(1) %8, align 256, !invariant.load !4
  %17 = load i64, ptr addrspace(1) %9, align 256, !invariant.load !4
  %18 = add i64 %17, %16
  %19 = icmp slt i64 %18, 0
  %20 = add i64 %18, 7
  %21 = select i1 %19, i64 %20, i64 %18
  %22 = trunc i64 %21 to i32
  %23 = tail call i32 @llvm.smax.i32(i32 %22, i32 0)
  %24 = tail call i32 @llvm.umin.i32(i32 %23, i32 6)
  %25 = call <4 x i32> @llvm.masked.load.v4i32.p1(ptr addrspace(1) align 256 %10, <4 x i1> <i1 true, i1 true, i1 true, i1 false>, <4 x i32> poison), !invariant.load !4
  %26 = extractelement <4 x i32> %25, i32 0
  %27 = extractelement <4 x i32> %25, i32 1
  %28 = extractelement <4 x i32> %25, i32 2
  %Extend8 = extractelement <4 x i32> %25, i32 3
  %29 = icmp sgt i32 %22, -1
  %30 = icmp sge i32 %26, %22
  %31 = and i1 %29, %30
  %32 = or i32 %28, %27
  %33 = icmp sgt i32 %32, -1
  %34 = and i1 %33, %31
  %35 = load double, ptr addrspace(1) %11, align 256, !invariant.load !4
  %36 = shl nuw nsw i32 %14, 7
  %37 = or disjoint i32 %36, %15
  %38 = zext nneg i32 %37 to i64
  %39 = getelementptr inbounds [16 x i8], ptr addrspace(1) %12, i64 %38
  %40 = load <2 x double>, ptr addrspace(1) %39, align 16, !invariant.load !4
  %.unpack9 = extractelement <2 x double> %40, i32 0
  %.unpack210 = extractelement <2 x double> %40, i32 1
  %41 = fmul double %35, %.unpack9
  %42 = fmul double %.unpack210, 0.000000e+00
  %43 = fmul double %.unpack9, 0.000000e+00
  %44 = fmul double %35, %.unpack210
  %45 = fsub double %41, %42
  %46 = fadd double %43, %44
  %47 = shl nuw nsw i32 %24, 12
  %48 = or disjoint i32 %47, %36
  %49 = or disjoint i32 %48, %15
  %50 = zext nneg i32 %49 to i64
  %51 = getelementptr inbounds [16 x i8], ptr addrspace(1) %13, i64 %50
  %52 = load <2 x double>, ptr addrspace(1) %51, align 16
  %.unpack311 = extractelement <2 x double> %52, i32 0
  %.unpack512 = extractelement <2 x double> %52, i32 1
  %.elt = select i1 %34, double %45, double %.unpack311
  %.elt7 = select i1 %34, double %46, double %.unpack512
  %53 = insertelement <2 x double> poison, double %.elt, i32 0
  %54 = insertelement <2 x double> %53, double %.elt7, i32 1
  store <2 x double> %54, ptr addrspace(1) %51, align 16
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smax.i32(i32, i32) #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.umin.i32(i32, i32) #3

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare <4 x i32> @llvm.masked.load.v4i32.p1(ptr addrspace(1) captures(none), <4 x i1>, <4 x i32>) #4

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "nvvm.reqntid"="128,1,1" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #4 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{i32 0, i32 32}
!3 = !{i32 0, i32 128}
!4 = !{}
