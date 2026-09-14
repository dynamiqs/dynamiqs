; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_reduce_fusion_1(ptr noalias nofree readonly align 256 captures(none) dereferenceable(458752) %0, ptr noalias nofree readonly align 256 captures(none) dereferenceable(112) %1, ptr noalias nofree writeonly align 256 captures(none) dereferenceable(65536) %2) local_unnamed_addr #0 {
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = addrspacecast ptr %0 to ptr addrspace(1)
  %6 = addrspacecast ptr %2 to ptr addrspace(1)
  %7 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !2
  %8 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !3
  %9 = load <2 x double>, ptr addrspace(1) %4, align 256, !invariant.load !4
  %.unpack44 = extractelement <2 x double> %9, i32 0
  %.unpack245 = extractelement <2 x double> %9, i32 1
  %10 = shl nuw nsw i32 %7, 7
  %11 = or disjoint i32 %10, %8
  %12 = zext nneg i32 %11 to i64
  %13 = getelementptr inbounds [16 x i8], ptr addrspace(1) %5, i64 %12
  %14 = load <2 x double>, ptr addrspace(1) %13, align 16, !invariant.load !4
  %.unpack358 = extractelement <2 x double> %14, i32 0
  %.unpack559 = extractelement <2 x double> %14, i32 1
  %15 = fmul double %.unpack44, %.unpack358
  %16 = fmul double %.unpack245, %.unpack559
  %17 = fmul double %.unpack245, %.unpack358
  %18 = fmul double %.unpack44, %.unpack559
  %19 = fsub double %15, %16
  %20 = fadd double %17, %18
  %21 = fadd double %19, 0.000000e+00
  %22 = fadd double %20, 0.000000e+00
  %23 = getelementptr inbounds i8, ptr addrspace(1) %4, i64 16
  %24 = load <2 x double>, ptr addrspace(1) %23, align 16, !invariant.load !4
  %.unpack646 = extractelement <2 x double> %24, i32 0
  %.unpack847 = extractelement <2 x double> %24, i32 1
  %25 = getelementptr inbounds i8, ptr addrspace(1) %13, i64 65536
  %26 = load <2 x double>, ptr addrspace(1) %25, align 16, !invariant.load !4
  %.unpack960 = extractelement <2 x double> %26, i32 0
  %.unpack1161 = extractelement <2 x double> %26, i32 1
  %27 = fmul double %.unpack646, %.unpack960
  %28 = fmul double %.unpack847, %.unpack1161
  %29 = fmul double %.unpack847, %.unpack960
  %30 = fmul double %.unpack646, %.unpack1161
  %31 = fsub double %27, %28
  %32 = fadd double %29, %30
  %33 = fadd double %21, %31
  %34 = fadd double %22, %32
  %35 = getelementptr inbounds i8, ptr addrspace(1) %4, i64 32
  %36 = load <2 x double>, ptr addrspace(1) %35, align 32, !invariant.load !4
  %.unpack1248 = extractelement <2 x double> %36, i32 0
  %.unpack1449 = extractelement <2 x double> %36, i32 1
  %37 = getelementptr inbounds i8, ptr addrspace(1) %13, i64 131072
  %38 = load <2 x double>, ptr addrspace(1) %37, align 16, !invariant.load !4
  %.unpack1562 = extractelement <2 x double> %38, i32 0
  %.unpack1763 = extractelement <2 x double> %38, i32 1
  %39 = fmul double %.unpack1248, %.unpack1562
  %40 = fmul double %.unpack1449, %.unpack1763
  %41 = fmul double %.unpack1449, %.unpack1562
  %42 = fmul double %.unpack1248, %.unpack1763
  %43 = fsub double %39, %40
  %44 = fadd double %41, %42
  %45 = fadd double %33, %43
  %46 = fadd double %34, %44
  %47 = getelementptr inbounds i8, ptr addrspace(1) %4, i64 48
  %48 = load <2 x double>, ptr addrspace(1) %47, align 16, !invariant.load !4
  %.unpack1850 = extractelement <2 x double> %48, i32 0
  %.unpack2051 = extractelement <2 x double> %48, i32 1
  %49 = getelementptr inbounds i8, ptr addrspace(1) %13, i64 196608
  %50 = load <2 x double>, ptr addrspace(1) %49, align 16, !invariant.load !4
  %.unpack2164 = extractelement <2 x double> %50, i32 0
  %.unpack2365 = extractelement <2 x double> %50, i32 1
  %51 = fmul double %.unpack1850, %.unpack2164
  %52 = fmul double %.unpack2051, %.unpack2365
  %53 = fmul double %.unpack2051, %.unpack2164
  %54 = fmul double %.unpack1850, %.unpack2365
  %55 = fsub double %51, %52
  %56 = fadd double %53, %54
  %57 = fadd double %45, %55
  %58 = fadd double %46, %56
  %59 = getelementptr inbounds i8, ptr addrspace(1) %4, i64 64
  %60 = load <2 x double>, ptr addrspace(1) %59, align 64, !invariant.load !4
  %.unpack2452 = extractelement <2 x double> %60, i32 0
  %.unpack2653 = extractelement <2 x double> %60, i32 1
  %61 = getelementptr inbounds i8, ptr addrspace(1) %13, i64 262144
  %62 = load <2 x double>, ptr addrspace(1) %61, align 16, !invariant.load !4
  %.unpack2766 = extractelement <2 x double> %62, i32 0
  %.unpack2967 = extractelement <2 x double> %62, i32 1
  %63 = fmul double %.unpack2452, %.unpack2766
  %64 = fmul double %.unpack2653, %.unpack2967
  %65 = fmul double %.unpack2653, %.unpack2766
  %66 = fmul double %.unpack2452, %.unpack2967
  %67 = fsub double %63, %64
  %68 = fadd double %65, %66
  %69 = fadd double %57, %67
  %70 = fadd double %58, %68
  %71 = getelementptr inbounds i8, ptr addrspace(1) %4, i64 80
  %72 = load <2 x double>, ptr addrspace(1) %71, align 16, !invariant.load !4
  %.unpack3054 = extractelement <2 x double> %72, i32 0
  %.unpack3255 = extractelement <2 x double> %72, i32 1
  %73 = getelementptr inbounds i8, ptr addrspace(1) %13, i64 327680
  %74 = load <2 x double>, ptr addrspace(1) %73, align 16, !invariant.load !4
  %.unpack3368 = extractelement <2 x double> %74, i32 0
  %.unpack3569 = extractelement <2 x double> %74, i32 1
  %75 = fmul double %.unpack3054, %.unpack3368
  %76 = fmul double %.unpack3255, %.unpack3569
  %77 = fmul double %.unpack3255, %.unpack3368
  %78 = fmul double %.unpack3054, %.unpack3569
  %79 = fsub double %75, %76
  %80 = fadd double %77, %78
  %81 = fadd double %69, %79
  %82 = fadd double %70, %80
  %83 = getelementptr inbounds i8, ptr addrspace(1) %4, i64 96
  %84 = load <2 x double>, ptr addrspace(1) %83, align 32, !invariant.load !4
  %.unpack3656 = extractelement <2 x double> %84, i32 0
  %.unpack3857 = extractelement <2 x double> %84, i32 1
  %85 = getelementptr inbounds i8, ptr addrspace(1) %13, i64 393216
  %86 = load <2 x double>, ptr addrspace(1) %85, align 16, !invariant.load !4
  %.unpack3970 = extractelement <2 x double> %86, i32 0
  %.unpack4171 = extractelement <2 x double> %86, i32 1
  %87 = fmul double %.unpack3656, %.unpack3970
  %88 = fmul double %.unpack3857, %.unpack4171
  %89 = fmul double %.unpack3857, %.unpack3970
  %90 = fmul double %.unpack3656, %.unpack4171
  %91 = fsub double %87, %88
  %92 = fadd double %89, %90
  %93 = fadd double %81, %91
  %94 = fadd double %82, %92
  %95 = getelementptr inbounds [16 x i8], ptr addrspace(1) %6, i64 %12
  %96 = insertelement <2 x double> poison, double %93, i32 0
  %97 = insertelement <2 x double> %96, double %94, i32 1
  store <2 x double> %97, ptr addrspace(1) %95, align 16
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
