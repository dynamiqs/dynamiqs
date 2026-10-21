; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_reduce_fusion(ptr noalias nofree readonly align 256 captures(none) dereferenceable(458752) %0, ptr noalias nofree readonly align 256 captures(none) dereferenceable(784) %1, ptr noalias nofree readonly align 256 captures(none) dereferenceable(8) %2, ptr noalias nofree readonly align 256 captures(none) dereferenceable(8) %3, ptr noalias nofree writeonly align 256 captures(none) dereferenceable(65536) %4) local_unnamed_addr #0 {
  %6 = addrspacecast ptr %2 to ptr addrspace(1)
  %7 = addrspacecast ptr %3 to ptr addrspace(1)
  %8 = addrspacecast ptr %1 to ptr addrspace(1)
  %9 = addrspacecast ptr %0 to ptr addrspace(1)
  %10 = addrspacecast ptr %4 to ptr addrspace(1)
  %11 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !2
  %12 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !3
  %13 = load i64, ptr addrspace(1) %6, align 256, !invariant.load !4
  %14 = load i64, ptr addrspace(1) %7, align 256, !invariant.load !4
  %15 = add i64 %14, %13
  %16 = trunc i64 %15 to i32
  %17 = icmp slt i32 %16, 0
  %18 = add i32 %16, 7
  %19 = select i1 %17, i32 %18, i32 %16
  %20 = tail call i32 @llvm.smax.i32(i32 %19, i32 0)
  %21 = tail call i32 @llvm.umin.i32(i32 %20, i32 6)
  %22 = mul nuw nsw i32 %21, 7
  %23 = zext nneg i32 %22 to i64
  %24 = getelementptr inbounds [16 x i8], ptr addrspace(1) %8, i64 %23
  %25 = load <2 x double>, ptr addrspace(1) %24, align 16, !invariant.load !4
  %.unpack44 = extractelement <2 x double> %25, i32 0
  %.unpack245 = extractelement <2 x double> %25, i32 1
  %26 = shl nuw nsw i32 %11, 7
  %27 = or disjoint i32 %26, %12
  %28 = zext nneg i32 %27 to i64
  %29 = getelementptr inbounds [16 x i8], ptr addrspace(1) %9, i64 %28
  %30 = load <2 x double>, ptr addrspace(1) %29, align 16, !invariant.load !4
  %.unpack358 = extractelement <2 x double> %30, i32 0
  %.unpack559 = extractelement <2 x double> %30, i32 1
  %31 = fmul double %.unpack44, %.unpack358
  %32 = fmul double %.unpack245, %.unpack559
  %33 = fmul double %.unpack245, %.unpack358
  %34 = fmul double %.unpack44, %.unpack559
  %35 = fsub double %31, %32
  %36 = fadd double %33, %34
  %37 = fadd double %35, 0.000000e+00
  %38 = fadd double %36, 0.000000e+00
  %39 = getelementptr inbounds i8, ptr addrspace(1) %24, i64 16
  %40 = load <2 x double>, ptr addrspace(1) %39, align 16, !invariant.load !4
  %.unpack646 = extractelement <2 x double> %40, i32 0
  %.unpack847 = extractelement <2 x double> %40, i32 1
  %41 = getelementptr inbounds i8, ptr addrspace(1) %29, i64 65536
  %42 = load <2 x double>, ptr addrspace(1) %41, align 16, !invariant.load !4
  %.unpack960 = extractelement <2 x double> %42, i32 0
  %.unpack1161 = extractelement <2 x double> %42, i32 1
  %43 = fmul double %.unpack646, %.unpack960
  %44 = fmul double %.unpack847, %.unpack1161
  %45 = fmul double %.unpack847, %.unpack960
  %46 = fmul double %.unpack646, %.unpack1161
  %47 = fsub double %43, %44
  %48 = fadd double %45, %46
  %49 = fadd double %37, %47
  %50 = fadd double %38, %48
  %51 = getelementptr inbounds i8, ptr addrspace(1) %24, i64 32
  %52 = load <2 x double>, ptr addrspace(1) %51, align 16, !invariant.load !4
  %.unpack1248 = extractelement <2 x double> %52, i32 0
  %.unpack1449 = extractelement <2 x double> %52, i32 1
  %53 = getelementptr inbounds i8, ptr addrspace(1) %29, i64 131072
  %54 = load <2 x double>, ptr addrspace(1) %53, align 16, !invariant.load !4
  %.unpack1562 = extractelement <2 x double> %54, i32 0
  %.unpack1763 = extractelement <2 x double> %54, i32 1
  %55 = fmul double %.unpack1248, %.unpack1562
  %56 = fmul double %.unpack1449, %.unpack1763
  %57 = fmul double %.unpack1449, %.unpack1562
  %58 = fmul double %.unpack1248, %.unpack1763
  %59 = fsub double %55, %56
  %60 = fadd double %57, %58
  %61 = fadd double %49, %59
  %62 = fadd double %50, %60
  %63 = getelementptr inbounds i8, ptr addrspace(1) %24, i64 48
  %64 = load <2 x double>, ptr addrspace(1) %63, align 16, !invariant.load !4
  %.unpack1850 = extractelement <2 x double> %64, i32 0
  %.unpack2051 = extractelement <2 x double> %64, i32 1
  %65 = getelementptr inbounds i8, ptr addrspace(1) %29, i64 196608
  %66 = load <2 x double>, ptr addrspace(1) %65, align 16, !invariant.load !4
  %.unpack2164 = extractelement <2 x double> %66, i32 0
  %.unpack2365 = extractelement <2 x double> %66, i32 1
  %67 = fmul double %.unpack1850, %.unpack2164
  %68 = fmul double %.unpack2051, %.unpack2365
  %69 = fmul double %.unpack2051, %.unpack2164
  %70 = fmul double %.unpack1850, %.unpack2365
  %71 = fsub double %67, %68
  %72 = fadd double %69, %70
  %73 = fadd double %61, %71
  %74 = fadd double %62, %72
  %75 = getelementptr inbounds i8, ptr addrspace(1) %24, i64 64
  %76 = load <2 x double>, ptr addrspace(1) %75, align 16, !invariant.load !4
  %.unpack2452 = extractelement <2 x double> %76, i32 0
  %.unpack2653 = extractelement <2 x double> %76, i32 1
  %77 = getelementptr inbounds i8, ptr addrspace(1) %29, i64 262144
  %78 = load <2 x double>, ptr addrspace(1) %77, align 16, !invariant.load !4
  %.unpack2766 = extractelement <2 x double> %78, i32 0
  %.unpack2967 = extractelement <2 x double> %78, i32 1
  %79 = fmul double %.unpack2452, %.unpack2766
  %80 = fmul double %.unpack2653, %.unpack2967
  %81 = fmul double %.unpack2653, %.unpack2766
  %82 = fmul double %.unpack2452, %.unpack2967
  %83 = fsub double %79, %80
  %84 = fadd double %81, %82
  %85 = fadd double %73, %83
  %86 = fadd double %74, %84
  %87 = getelementptr inbounds i8, ptr addrspace(1) %24, i64 80
  %88 = load <2 x double>, ptr addrspace(1) %87, align 16, !invariant.load !4
  %.unpack3054 = extractelement <2 x double> %88, i32 0
  %.unpack3255 = extractelement <2 x double> %88, i32 1
  %89 = getelementptr inbounds i8, ptr addrspace(1) %29, i64 327680
  %90 = load <2 x double>, ptr addrspace(1) %89, align 16, !invariant.load !4
  %.unpack3368 = extractelement <2 x double> %90, i32 0
  %.unpack3569 = extractelement <2 x double> %90, i32 1
  %91 = fmul double %.unpack3054, %.unpack3368
  %92 = fmul double %.unpack3255, %.unpack3569
  %93 = fmul double %.unpack3255, %.unpack3368
  %94 = fmul double %.unpack3054, %.unpack3569
  %95 = fsub double %91, %92
  %96 = fadd double %93, %94
  %97 = fadd double %85, %95
  %98 = fadd double %86, %96
  %99 = getelementptr inbounds i8, ptr addrspace(1) %24, i64 96
  %100 = load <2 x double>, ptr addrspace(1) %99, align 16, !invariant.load !4
  %.unpack3656 = extractelement <2 x double> %100, i32 0
  %.unpack3857 = extractelement <2 x double> %100, i32 1
  %101 = getelementptr inbounds i8, ptr addrspace(1) %29, i64 393216
  %102 = load <2 x double>, ptr addrspace(1) %101, align 16, !invariant.load !4
  %.unpack3970 = extractelement <2 x double> %102, i32 0
  %.unpack4171 = extractelement <2 x double> %102, i32 1
  %103 = fmul double %.unpack3656, %.unpack3970
  %104 = fmul double %.unpack3857, %.unpack4171
  %105 = fmul double %.unpack3857, %.unpack3970
  %106 = fmul double %.unpack3656, %.unpack4171
  %107 = fsub double %103, %104
  %108 = fadd double %105, %106
  %109 = fadd double %97, %107
  %110 = fadd double %98, %108
  %111 = getelementptr inbounds [16 x i8], ptr addrspace(1) %10, i64 %28
  %112 = insertelement <2 x double> poison, double %109, i32 0
  %113 = insertelement <2 x double> %112, double %110, i32 1
  store <2 x double> %113, ptr addrspace(1) %111, align 16
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

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "nvvm.reqntid"="128,1,1" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{i32 0, i32 32}
!3 = !{i32 0, i32 128}
!4 = !{}
