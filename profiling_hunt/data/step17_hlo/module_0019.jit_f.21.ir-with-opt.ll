; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_add_fusion_1(ptr noalias nofree readonly align 256 captures(none) dereferenceable(65536) %0, ptr noalias nofree readonly align 256 captures(none) dereferenceable(458752) %1, ptr noalias nofree readonly align 256 captures(none) dereferenceable(784) %2, ptr noalias nofree readonly align 256 captures(none) dereferenceable(8) %3, ptr noalias nofree readonly align 256 captures(none) dereferenceable(8) %4, ptr noalias nofree writeonly align 256 captures(none) dereferenceable(65536) %5) local_unnamed_addr #0 {
  %7 = addrspacecast ptr %0 to ptr addrspace(1)
  %8 = addrspacecast ptr %3 to ptr addrspace(1)
  %9 = addrspacecast ptr %4 to ptr addrspace(1)
  %10 = addrspacecast ptr %2 to ptr addrspace(1)
  %11 = addrspacecast ptr %1 to ptr addrspace(1)
  %12 = addrspacecast ptr %5 to ptr addrspace(1)
  %13 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !2
  %14 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !3
  %15 = shl nuw nsw i32 %13, 7
  %16 = or disjoint i32 %15, %14
  %17 = zext nneg i32 %16 to i64
  %18 = getelementptr inbounds [16 x i8], ptr addrspace(1) %7, i64 %17
  %19 = load <2 x double>, ptr addrspace(1) %18, align 16, !invariant.load !4
  %.unpack47 = extractelement <2 x double> %19, i32 0
  %.unpack248 = extractelement <2 x double> %19, i32 1
  %20 = load i64, ptr addrspace(1) %8, align 256, !invariant.load !4
  %21 = load i64, ptr addrspace(1) %9, align 256, !invariant.load !4
  %22 = add i64 %21, %20
  %23 = trunc i64 %22 to i32
  %24 = icmp slt i32 %23, 0
  %25 = add i32 %23, 7
  %26 = select i1 %24, i32 %25, i32 %23
  %27 = tail call i32 @llvm.smax.i32(i32 %26, i32 0)
  %28 = tail call i32 @llvm.umin.i32(i32 %27, i32 6)
  %29 = mul nuw nsw i32 %28, 7
  %30 = zext nneg i32 %29 to i64
  %31 = getelementptr inbounds [16 x i8], ptr addrspace(1) %10, i64 %30
  %32 = load <2 x double>, ptr addrspace(1) %31, align 16, !invariant.load !4
  %.unpack349 = extractelement <2 x double> %32, i32 0
  %.unpack550 = extractelement <2 x double> %32, i32 1
  %33 = getelementptr inbounds [16 x i8], ptr addrspace(1) %11, i64 %17
  %34 = load <2 x double>, ptr addrspace(1) %33, align 16, !invariant.load !4
  %.unpack663 = extractelement <2 x double> %34, i32 0
  %.unpack864 = extractelement <2 x double> %34, i32 1
  %35 = fmul double %.unpack349, %.unpack663
  %36 = fmul double %.unpack550, %.unpack864
  %37 = fmul double %.unpack550, %.unpack663
  %38 = fmul double %.unpack349, %.unpack864
  %39 = fsub double %35, %36
  %40 = fadd double %37, %38
  %41 = fadd double %39, 0.000000e+00
  %42 = fadd double %40, 0.000000e+00
  %43 = getelementptr inbounds i8, ptr addrspace(1) %31, i64 16
  %44 = load <2 x double>, ptr addrspace(1) %43, align 16, !invariant.load !4
  %.unpack951 = extractelement <2 x double> %44, i32 0
  %.unpack1152 = extractelement <2 x double> %44, i32 1
  %45 = getelementptr inbounds i8, ptr addrspace(1) %33, i64 65536
  %46 = load <2 x double>, ptr addrspace(1) %45, align 16, !invariant.load !4
  %.unpack1265 = extractelement <2 x double> %46, i32 0
  %.unpack1466 = extractelement <2 x double> %46, i32 1
  %47 = fmul double %.unpack951, %.unpack1265
  %48 = fmul double %.unpack1152, %.unpack1466
  %49 = fmul double %.unpack1152, %.unpack1265
  %50 = fmul double %.unpack951, %.unpack1466
  %51 = fsub double %47, %48
  %52 = fadd double %49, %50
  %53 = fadd double %41, %51
  %54 = fadd double %42, %52
  %55 = getelementptr inbounds i8, ptr addrspace(1) %31, i64 32
  %56 = load <2 x double>, ptr addrspace(1) %55, align 16, !invariant.load !4
  %.unpack1553 = extractelement <2 x double> %56, i32 0
  %.unpack1754 = extractelement <2 x double> %56, i32 1
  %57 = getelementptr inbounds i8, ptr addrspace(1) %33, i64 131072
  %58 = load <2 x double>, ptr addrspace(1) %57, align 16, !invariant.load !4
  %.unpack1867 = extractelement <2 x double> %58, i32 0
  %.unpack2068 = extractelement <2 x double> %58, i32 1
  %59 = fmul double %.unpack1553, %.unpack1867
  %60 = fmul double %.unpack1754, %.unpack2068
  %61 = fmul double %.unpack1754, %.unpack1867
  %62 = fmul double %.unpack1553, %.unpack2068
  %63 = fsub double %59, %60
  %64 = fadd double %61, %62
  %65 = fadd double %53, %63
  %66 = fadd double %54, %64
  %67 = getelementptr inbounds i8, ptr addrspace(1) %31, i64 48
  %68 = load <2 x double>, ptr addrspace(1) %67, align 16, !invariant.load !4
  %.unpack2155 = extractelement <2 x double> %68, i32 0
  %.unpack2356 = extractelement <2 x double> %68, i32 1
  %69 = getelementptr inbounds i8, ptr addrspace(1) %33, i64 196608
  %70 = load <2 x double>, ptr addrspace(1) %69, align 16, !invariant.load !4
  %.unpack2469 = extractelement <2 x double> %70, i32 0
  %.unpack2670 = extractelement <2 x double> %70, i32 1
  %71 = fmul double %.unpack2155, %.unpack2469
  %72 = fmul double %.unpack2356, %.unpack2670
  %73 = fmul double %.unpack2356, %.unpack2469
  %74 = fmul double %.unpack2155, %.unpack2670
  %75 = fsub double %71, %72
  %76 = fadd double %73, %74
  %77 = fadd double %65, %75
  %78 = fadd double %66, %76
  %79 = getelementptr inbounds i8, ptr addrspace(1) %31, i64 64
  %80 = load <2 x double>, ptr addrspace(1) %79, align 16, !invariant.load !4
  %.unpack2757 = extractelement <2 x double> %80, i32 0
  %.unpack2958 = extractelement <2 x double> %80, i32 1
  %81 = getelementptr inbounds i8, ptr addrspace(1) %33, i64 262144
  %82 = load <2 x double>, ptr addrspace(1) %81, align 16, !invariant.load !4
  %.unpack3071 = extractelement <2 x double> %82, i32 0
  %.unpack3272 = extractelement <2 x double> %82, i32 1
  %83 = fmul double %.unpack2757, %.unpack3071
  %84 = fmul double %.unpack2958, %.unpack3272
  %85 = fmul double %.unpack2958, %.unpack3071
  %86 = fmul double %.unpack2757, %.unpack3272
  %87 = fsub double %83, %84
  %88 = fadd double %85, %86
  %89 = fadd double %77, %87
  %90 = fadd double %78, %88
  %91 = getelementptr inbounds i8, ptr addrspace(1) %31, i64 80
  %92 = load <2 x double>, ptr addrspace(1) %91, align 16, !invariant.load !4
  %.unpack3359 = extractelement <2 x double> %92, i32 0
  %.unpack3560 = extractelement <2 x double> %92, i32 1
  %93 = getelementptr inbounds i8, ptr addrspace(1) %33, i64 327680
  %94 = load <2 x double>, ptr addrspace(1) %93, align 16, !invariant.load !4
  %.unpack3673 = extractelement <2 x double> %94, i32 0
  %.unpack3874 = extractelement <2 x double> %94, i32 1
  %95 = fmul double %.unpack3359, %.unpack3673
  %96 = fmul double %.unpack3560, %.unpack3874
  %97 = fmul double %.unpack3560, %.unpack3673
  %98 = fmul double %.unpack3359, %.unpack3874
  %99 = fsub double %95, %96
  %100 = fadd double %97, %98
  %101 = fadd double %89, %99
  %102 = fadd double %90, %100
  %103 = getelementptr inbounds i8, ptr addrspace(1) %31, i64 96
  %104 = load <2 x double>, ptr addrspace(1) %103, align 16, !invariant.load !4
  %.unpack3961 = extractelement <2 x double> %104, i32 0
  %.unpack4162 = extractelement <2 x double> %104, i32 1
  %105 = getelementptr inbounds i8, ptr addrspace(1) %33, i64 393216
  %106 = load <2 x double>, ptr addrspace(1) %105, align 16, !invariant.load !4
  %.unpack4275 = extractelement <2 x double> %106, i32 0
  %.unpack4476 = extractelement <2 x double> %106, i32 1
  %107 = fmul double %.unpack3961, %.unpack4275
  %108 = fmul double %.unpack4162, %.unpack4476
  %109 = fmul double %.unpack4162, %.unpack4275
  %110 = fmul double %.unpack3961, %.unpack4476
  %111 = fsub double %107, %108
  %112 = fadd double %109, %110
  %113 = fadd double %101, %111
  %114 = fadd double %102, %112
  %115 = fadd double %.unpack47, %113
  %116 = fadd double %.unpack248, %114
  %117 = getelementptr inbounds [16 x i8], ptr addrspace(1) %12, i64 %17
  %118 = insertelement <2 x double> poison, double %115, i32 0
  %119 = insertelement <2 x double> %118, double %116, i32 1
  store <2 x double> %119, ptr addrspace(1) %117, align 16
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
