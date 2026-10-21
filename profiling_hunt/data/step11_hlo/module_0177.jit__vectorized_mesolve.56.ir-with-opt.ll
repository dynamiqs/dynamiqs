; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@shared_0 = private unnamed_addr addrspace(3) global [1056 x double] undef

; Function Attrs: norecurse nounwind
define ptx_kernel void @input_transpose_fusion_2(ptr noalias nofree readonly align 256 captures(none) dereferenceable(65536) %0, ptr noalias nofree writeonly align 256 captures(none) dereferenceable(32768) %1) local_unnamed_addr #0 {
  %3 = addrspacecast ptr %0 to ptr addrspace(1)
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !4
  %6 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !5
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
  %19 = load <2 x double>, ptr addrspace(1) %18, align 16, !invariant.load !6
  %.unpack24 = extractelement <2 x double> %19, i32 0
  %.unpack225 = extractelement <2 x double> %19, i32 1
  %20 = tail call double @llvm.nvvm.fabs.f64(double %.unpack24)
  %21 = tail call double @llvm.nvvm.fabs.f64(double %.unpack225)
  %22 = tail call double @llvm.maximum.f64(double %20, double %21)
  %23 = tail call double @llvm.minimum.f64(double %20, double %21)
  %24 = fdiv double %23, %22
  %25 = fmul double %24, %24
  %26 = fadd double %25, 1.000000e+00
  %27 = tail call double @llvm.sqrt.f64(double %26)
  %28 = fmul double %22, %27
  %29 = fcmp uno double %28, 0.000000e+00
  %30 = select i1 %29, double %23, double %28
  %31 = mul nuw nsw i32 %15, 33
  %32 = add nuw nsw i32 %31, %9
  %33 = zext nneg i32 %32 to i64
  %34 = getelementptr inbounds [8 x i8], ptr addrspace(3) @shared_0, i64 %33
  store double %30, ptr addrspace(3) %34, align 8
  %35 = getelementptr inbounds i8, ptr addrspace(1) %18, i64 4096
  %36 = load <2 x double>, ptr addrspace(1) %35, align 16, !invariant.load !6
  %.unpack326 = extractelement <2 x double> %36, i32 0
  %.unpack527 = extractelement <2 x double> %36, i32 1
  %37 = tail call double @llvm.nvvm.fabs.f64(double %.unpack326)
  %38 = tail call double @llvm.nvvm.fabs.f64(double %.unpack527)
  %39 = tail call double @llvm.maximum.f64(double %37, double %38)
  %40 = tail call double @llvm.minimum.f64(double %37, double %38)
  %41 = fdiv double %40, %39
  %42 = fmul double %41, %41
  %43 = fadd double %42, 1.000000e+00
  %44 = tail call double @llvm.sqrt.f64(double %43)
  %45 = fmul double %39, %44
  %46 = fcmp uno double %45, 0.000000e+00
  %47 = select i1 %46, double %40, double %45
  %48 = getelementptr inbounds i8, ptr addrspace(3) %34, i64 32
  store double %47, ptr addrspace(3) %48, align 8
  %49 = getelementptr inbounds i8, ptr addrspace(1) %18, i64 8192
  %50 = load <2 x double>, ptr addrspace(1) %49, align 16, !invariant.load !6
  %.unpack628 = extractelement <2 x double> %50, i32 0
  %.unpack829 = extractelement <2 x double> %50, i32 1
  %51 = tail call double @llvm.nvvm.fabs.f64(double %.unpack628)
  %52 = tail call double @llvm.nvvm.fabs.f64(double %.unpack829)
  %53 = tail call double @llvm.maximum.f64(double %51, double %52)
  %54 = tail call double @llvm.minimum.f64(double %51, double %52)
  %55 = fdiv double %54, %53
  %56 = fmul double %55, %55
  %57 = fadd double %56, 1.000000e+00
  %58 = tail call double @llvm.sqrt.f64(double %57)
  %59 = fmul double %53, %58
  %60 = fcmp uno double %59, 0.000000e+00
  %61 = select i1 %60, double %54, double %59
  %62 = getelementptr inbounds i8, ptr addrspace(3) %34, i64 64
  store double %61, ptr addrspace(3) %62, align 8
  %63 = getelementptr inbounds i8, ptr addrspace(1) %18, i64 12288
  %64 = load <2 x double>, ptr addrspace(1) %63, align 16, !invariant.load !6
  %.unpack930 = extractelement <2 x double> %64, i32 0
  %.unpack1131 = extractelement <2 x double> %64, i32 1
  %65 = tail call double @llvm.nvvm.fabs.f64(double %.unpack930)
  %66 = tail call double @llvm.nvvm.fabs.f64(double %.unpack1131)
  %67 = tail call double @llvm.maximum.f64(double %65, double %66)
  %68 = tail call double @llvm.minimum.f64(double %65, double %66)
  %69 = fdiv double %68, %67
  %70 = fmul double %69, %69
  %71 = fadd double %70, 1.000000e+00
  %72 = tail call double @llvm.sqrt.f64(double %71)
  %73 = fmul double %67, %72
  %74 = fcmp uno double %73, 0.000000e+00
  %75 = select i1 %74, double %68, double %73
  %76 = getelementptr inbounds i8, ptr addrspace(3) %34, i64 96
  store double %75, ptr addrspace(3) %76, align 8
  %77 = getelementptr inbounds i8, ptr addrspace(1) %18, i64 16384
  %78 = load <2 x double>, ptr addrspace(1) %77, align 16, !invariant.load !6
  %.unpack1232 = extractelement <2 x double> %78, i32 0
  %.unpack1433 = extractelement <2 x double> %78, i32 1
  %79 = tail call double @llvm.nvvm.fabs.f64(double %.unpack1232)
  %80 = tail call double @llvm.nvvm.fabs.f64(double %.unpack1433)
  %81 = tail call double @llvm.maximum.f64(double %79, double %80)
  %82 = tail call double @llvm.minimum.f64(double %79, double %80)
  %83 = fdiv double %82, %81
  %84 = fmul double %83, %83
  %85 = fadd double %84, 1.000000e+00
  %86 = tail call double @llvm.sqrt.f64(double %85)
  %87 = fmul double %81, %86
  %88 = fcmp uno double %87, 0.000000e+00
  %89 = select i1 %88, double %82, double %87
  %90 = getelementptr inbounds i8, ptr addrspace(3) %34, i64 128
  store double %89, ptr addrspace(3) %90, align 8
  %91 = getelementptr inbounds i8, ptr addrspace(1) %18, i64 20480
  %92 = load <2 x double>, ptr addrspace(1) %91, align 16, !invariant.load !6
  %.unpack1534 = extractelement <2 x double> %92, i32 0
  %.unpack1735 = extractelement <2 x double> %92, i32 1
  %93 = tail call double @llvm.nvvm.fabs.f64(double %.unpack1534)
  %94 = tail call double @llvm.nvvm.fabs.f64(double %.unpack1735)
  %95 = tail call double @llvm.maximum.f64(double %93, double %94)
  %96 = tail call double @llvm.minimum.f64(double %93, double %94)
  %97 = fdiv double %96, %95
  %98 = fmul double %97, %97
  %99 = fadd double %98, 1.000000e+00
  %100 = tail call double @llvm.sqrt.f64(double %99)
  %101 = fmul double %95, %100
  %102 = fcmp uno double %101, 0.000000e+00
  %103 = select i1 %102, double %96, double %101
  %104 = getelementptr inbounds i8, ptr addrspace(3) %34, i64 160
  store double %103, ptr addrspace(3) %104, align 8
  %105 = getelementptr inbounds i8, ptr addrspace(1) %18, i64 24576
  %106 = load <2 x double>, ptr addrspace(1) %105, align 16, !invariant.load !6
  %.unpack1836 = extractelement <2 x double> %106, i32 0
  %.unpack2037 = extractelement <2 x double> %106, i32 1
  %107 = tail call double @llvm.nvvm.fabs.f64(double %.unpack1836)
  %108 = tail call double @llvm.nvvm.fabs.f64(double %.unpack2037)
  %109 = tail call double @llvm.maximum.f64(double %107, double %108)
  %110 = tail call double @llvm.minimum.f64(double %107, double %108)
  %111 = fdiv double %110, %109
  %112 = fmul double %111, %111
  %113 = fadd double %112, 1.000000e+00
  %114 = tail call double @llvm.sqrt.f64(double %113)
  %115 = fmul double %109, %114
  %116 = fcmp uno double %115, 0.000000e+00
  %117 = select i1 %116, double %110, double %115
  %118 = getelementptr inbounds i8, ptr addrspace(3) %34, i64 192
  store double %117, ptr addrspace(3) %118, align 8
  %119 = getelementptr inbounds i8, ptr addrspace(1) %18, i64 28672
  %120 = load <2 x double>, ptr addrspace(1) %119, align 16, !invariant.load !6
  %.unpack2138 = extractelement <2 x double> %120, i32 0
  %.unpack2339 = extractelement <2 x double> %120, i32 1
  %121 = tail call double @llvm.nvvm.fabs.f64(double %.unpack2138)
  %122 = tail call double @llvm.nvvm.fabs.f64(double %.unpack2339)
  %123 = tail call double @llvm.maximum.f64(double %121, double %122)
  %124 = tail call double @llvm.minimum.f64(double %121, double %122)
  %125 = fdiv double %124, %123
  %126 = fmul double %125, %125
  %127 = fadd double %126, 1.000000e+00
  %128 = tail call double @llvm.sqrt.f64(double %127)
  %129 = fmul double %123, %128
  %130 = fcmp uno double %129, 0.000000e+00
  %131 = select i1 %130, double %124, double %129
  %132 = getelementptr inbounds i8, ptr addrspace(3) %34, i64 224
  store double %131, ptr addrspace(3) %132, align 8
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %133 = mul nuw nsw i32 %9, 33
  %134 = add nuw nsw i32 %133, %15
  %135 = zext nneg i32 %134 to i64
  %136 = getelementptr inbounds [8 x i8], ptr addrspace(3) @shared_0, i64 %135
  %137 = load double, ptr addrspace(3) %136, align 8
  %138 = shl nuw nsw i32 %7, 11
  %139 = shl nuw nsw i32 %11, 5
  %140 = or disjoint i32 %138, %139
  %141 = or disjoint i32 %140, %10
  %142 = or disjoint i32 %141, %15
  %143 = zext nneg i32 %142 to i64
  %144 = getelementptr inbounds [8 x i8], ptr addrspace(1) %4, i64 %143
  store double %137, ptr addrspace(1) %144, align 8
  %145 = getelementptr inbounds i8, ptr addrspace(3) %136, i64 1056
  %146 = load double, ptr addrspace(3) %145, align 8
  %147 = getelementptr inbounds i8, ptr addrspace(1) %144, i64 2048
  store double %146, ptr addrspace(1) %147, align 8
  %148 = getelementptr inbounds i8, ptr addrspace(3) %136, i64 2112
  %149 = load double, ptr addrspace(3) %148, align 8
  %150 = getelementptr inbounds i8, ptr addrspace(1) %144, i64 4096
  store double %149, ptr addrspace(1) %150, align 8
  %151 = getelementptr inbounds i8, ptr addrspace(3) %136, i64 3168
  %152 = load double, ptr addrspace(3) %151, align 8
  %153 = getelementptr inbounds i8, ptr addrspace(1) %144, i64 6144
  store double %152, ptr addrspace(1) %153, align 8
  %154 = getelementptr inbounds i8, ptr addrspace(3) %136, i64 4224
  %155 = load double, ptr addrspace(3) %154, align 8
  %156 = getelementptr inbounds i8, ptr addrspace(1) %144, i64 8192
  store double %155, ptr addrspace(1) %156, align 8
  %157 = getelementptr inbounds i8, ptr addrspace(3) %136, i64 5280
  %158 = load double, ptr addrspace(3) %157, align 8
  %159 = getelementptr inbounds i8, ptr addrspace(1) %144, i64 10240
  store double %158, ptr addrspace(1) %159, align 8
  %160 = getelementptr inbounds i8, ptr addrspace(3) %136, i64 6336
  %161 = load double, ptr addrspace(3) %160, align 8
  %162 = getelementptr inbounds i8, ptr addrspace(1) %144, i64 12288
  store double %161, ptr addrspace(1) %162, align 8
  %163 = getelementptr inbounds i8, ptr addrspace(3) %136, i64 7392
  %164 = load double, ptr addrspace(3) %163, align 8
  %165 = getelementptr inbounds i8, ptr addrspace(1) %144, i64 14336
  store double %164, ptr addrspace(1) %165, align 8
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.maximum.f64(double, double) #2

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.minimum.f64(double, double) #2

; Function Attrs: convergent nocallback nounwind
declare void @llvm.nvvm.barrier.cta.sync.aligned.all(i32) #3

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.nvvm.fabs.f64(double) #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.sqrt.f64(double) #4

attributes #0 = { norecurse nounwind "nvvm.reqntid"="128,1,1" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { convergent nocallback nounwind }
attributes #4 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}
!nvvmir.version = !{!3}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}
!3 = !{i32 2, i32 0}
!4 = !{i32 0, i32 128}
!5 = !{i32 0, i32 4}
!6 = !{}
