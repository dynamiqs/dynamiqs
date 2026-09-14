; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_dynamic_update_slice_fusion(ptr noalias nofree align 256 captures(none) dereferenceable(10485760) %0, ptr noalias nofree readonly align 256 captures(none) dereferenceable(65536) %1, ptr noalias nofree readonly align 256 captures(none) dereferenceable(458752) %2, ptr noalias nofree readonly align 256 captures(none) dereferenceable(56) %3, ptr noalias nofree readonly align 256 captures(none) dereferenceable(8) %4, ptr noalias nofree readonly align 256 captures(none) dereferenceable(8) %5, ptr noalias nofree readonly align 256 captures(none) dereferenceable(12) %6, ptr noalias nofree readnone align 256 captures(none) dereferenceable(10485760) %7) local_unnamed_addr #0 {
  %9 = addrspacecast ptr %4 to ptr addrspace(1)
  %10 = addrspacecast ptr %5 to ptr addrspace(1)
  %11 = addrspacecast ptr %6 to ptr addrspace(1)
  %12 = addrspacecast ptr %1 to ptr addrspace(1)
  %13 = addrspacecast ptr %3 to ptr addrspace(1)
  %14 = addrspacecast ptr %2 to ptr addrspace(1)
  %15 = addrspacecast ptr %0 to ptr addrspace(1)
  %16 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !2
  %17 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !3
  %18 = load i64, ptr addrspace(1) %9, align 256, !invariant.load !4
  %19 = load i64, ptr addrspace(1) %10, align 256, !invariant.load !4
  %20 = add i64 %19, %18
  %21 = icmp slt i64 %20, 0
  %22 = add i64 %20, 160
  %23 = select i1 %21, i64 %22, i64 %20
  %24 = trunc i64 %23 to i32
  %25 = tail call i32 @llvm.smax.i32(i32 %24, i32 0)
  %26 = tail call i32 @llvm.umin.i32(i32 %25, i32 159)
  %27 = call <4 x i32> @llvm.masked.load.v4i32.p1(ptr addrspace(1) align 256 %11, <4 x i1> <i1 true, i1 true, i1 true, i1 false>, <4 x i32> poison), !invariant.load !4
  %28 = extractelement <4 x i32> %27, i32 0
  %29 = extractelement <4 x i32> %27, i32 1
  %30 = extractelement <4 x i32> %27, i32 2
  %Extend29 = extractelement <4 x i32> %27, i32 3
  %31 = icmp sgt i32 %24, -1
  %32 = icmp sge i32 %28, %24
  %33 = and i1 %31, %32
  %34 = or i32 %30, %29
  %35 = icmp sgt i32 %34, -1
  %36 = and i1 %35, %33
  %37 = shl nuw nsw i32 %16, 7
  %38 = or disjoint i32 %37, %17
  %39 = zext nneg i32 %38 to i64
  %40 = getelementptr inbounds [16 x i8], ptr addrspace(1) %12, i64 %39
  %41 = load <2 x double>, ptr addrspace(1) %40, align 16, !invariant.load !4
  %.unpack30 = extractelement <2 x double> %41, i32 0
  %.unpack231 = extractelement <2 x double> %41, i32 1
  %42 = load <2 x double>, ptr addrspace(1) %13, align 256, !invariant.load !4
  %43 = extractelement <2 x double> %42, i32 0
  %44 = extractelement <2 x double> %42, i32 1
  %45 = getelementptr inbounds [16 x i8], ptr addrspace(1) %14, i64 %39
  %46 = load <2 x double>, ptr addrspace(1) %45, align 16, !invariant.load !4
  %.unpack332 = extractelement <2 x double> %46, i32 0
  %.unpack533 = extractelement <2 x double> %46, i32 1
  %47 = fmul double %43, %.unpack332
  %48 = fmul double %.unpack533, 0.000000e+00
  %49 = fmul double %.unpack332, 0.000000e+00
  %50 = fmul double %43, %.unpack533
  %51 = fsub double %47, %48
  %52 = fadd double %49, %50
  %53 = fadd double %51, 0.000000e+00
  %54 = fadd double %52, 0.000000e+00
  %55 = getelementptr inbounds i8, ptr addrspace(1) %45, i64 65536
  %56 = load <2 x double>, ptr addrspace(1) %55, align 16, !invariant.load !4
  %.unpack634 = extractelement <2 x double> %56, i32 0
  %.unpack835 = extractelement <2 x double> %56, i32 1
  %57 = fmul double %44, %.unpack634
  %58 = fmul double %.unpack835, 0.000000e+00
  %59 = fmul double %.unpack634, 0.000000e+00
  %60 = fmul double %44, %.unpack835
  %61 = fsub double %57, %58
  %62 = fadd double %59, %60
  %63 = fadd double %53, %61
  %64 = fadd double %54, %62
  %65 = getelementptr inbounds i8, ptr addrspace(1) %13, i64 16
  %66 = load <2 x double>, ptr addrspace(1) %65, align 16, !invariant.load !4
  %67 = extractelement <2 x double> %66, i32 0
  %68 = extractelement <2 x double> %66, i32 1
  %69 = getelementptr inbounds i8, ptr addrspace(1) %45, i64 131072
  %70 = load <2 x double>, ptr addrspace(1) %69, align 16, !invariant.load !4
  %.unpack936 = extractelement <2 x double> %70, i32 0
  %.unpack1137 = extractelement <2 x double> %70, i32 1
  %71 = fmul double %67, %.unpack936
  %72 = fmul double %.unpack1137, 0.000000e+00
  %73 = fmul double %.unpack936, 0.000000e+00
  %74 = fmul double %67, %.unpack1137
  %75 = fsub double %71, %72
  %76 = fadd double %73, %74
  %77 = fadd double %63, %75
  %78 = fadd double %64, %76
  %79 = getelementptr inbounds i8, ptr addrspace(1) %45, i64 196608
  %80 = load <2 x double>, ptr addrspace(1) %79, align 16, !invariant.load !4
  %.unpack1238 = extractelement <2 x double> %80, i32 0
  %.unpack1439 = extractelement <2 x double> %80, i32 1
  %81 = fmul double %68, %.unpack1238
  %82 = fmul double %.unpack1439, 0.000000e+00
  %83 = fmul double %.unpack1238, 0.000000e+00
  %84 = fmul double %68, %.unpack1439
  %85 = fsub double %81, %82
  %86 = fadd double %83, %84
  %87 = fadd double %77, %85
  %88 = fadd double %78, %86
  %89 = getelementptr inbounds i8, ptr addrspace(1) %13, i64 32
  %90 = load <2 x double>, ptr addrspace(1) %89, align 32, !invariant.load !4
  %91 = extractelement <2 x double> %90, i32 0
  %92 = extractelement <2 x double> %90, i32 1
  %93 = getelementptr inbounds i8, ptr addrspace(1) %45, i64 262144
  %94 = load <2 x double>, ptr addrspace(1) %93, align 16, !invariant.load !4
  %.unpack1540 = extractelement <2 x double> %94, i32 0
  %.unpack1741 = extractelement <2 x double> %94, i32 1
  %95 = fmul double %91, %.unpack1540
  %96 = fmul double %.unpack1741, 0.000000e+00
  %97 = fmul double %.unpack1540, 0.000000e+00
  %98 = fmul double %91, %.unpack1741
  %99 = fsub double %95, %96
  %100 = fadd double %97, %98
  %101 = fadd double %87, %99
  %102 = fadd double %88, %100
  %103 = getelementptr inbounds i8, ptr addrspace(1) %45, i64 327680
  %104 = load <2 x double>, ptr addrspace(1) %103, align 16, !invariant.load !4
  %.unpack1842 = extractelement <2 x double> %104, i32 0
  %.unpack2043 = extractelement <2 x double> %104, i32 1
  %105 = fmul double %92, %.unpack1842
  %106 = fmul double %.unpack2043, 0.000000e+00
  %107 = fmul double %.unpack1842, 0.000000e+00
  %108 = fmul double %92, %.unpack2043
  %109 = fsub double %105, %106
  %110 = fadd double %107, %108
  %111 = fadd double %101, %109
  %112 = fadd double %102, %110
  %113 = getelementptr inbounds i8, ptr addrspace(1) %13, i64 48
  %114 = load double, ptr addrspace(1) %113, align 16, !invariant.load !4
  %115 = getelementptr inbounds i8, ptr addrspace(1) %45, i64 393216
  %116 = load <2 x double>, ptr addrspace(1) %115, align 16, !invariant.load !4
  %.unpack2144 = extractelement <2 x double> %116, i32 0
  %.unpack2345 = extractelement <2 x double> %116, i32 1
  %117 = fmul double %114, %.unpack2144
  %118 = fmul double %.unpack2345, 0.000000e+00
  %119 = fmul double %.unpack2144, 0.000000e+00
  %120 = fmul double %114, %.unpack2345
  %121 = fsub double %117, %118
  %122 = fadd double %119, %120
  %123 = fadd double %111, %121
  %124 = fadd double %112, %122
  %125 = fadd double %.unpack30, %123
  %126 = fadd double %.unpack231, %124
  %127 = shl nuw nsw i32 %26, 12
  %128 = or disjoint i32 %127, %37
  %129 = or disjoint i32 %128, %17
  %130 = zext nneg i32 %129 to i64
  %131 = getelementptr inbounds [16 x i8], ptr addrspace(1) %15, i64 %130
  %132 = load <2 x double>, ptr addrspace(1) %131, align 16
  %.unpack2446 = extractelement <2 x double> %132, i32 0
  %.unpack2647 = extractelement <2 x double> %132, i32 1
  %.elt = select i1 %36, double %125, double %.unpack2446
  %.elt28 = select i1 %36, double %126, double %.unpack2647
  %133 = insertelement <2 x double> poison, double %.elt, i32 0
  %134 = insertelement <2 x double> %133, double %.elt28, i32 1
  store <2 x double> %134, ptr addrspace(1) %131, align 16
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
