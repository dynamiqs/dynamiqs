; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_multiply_fusion(ptr noalias nofree readonly align 256 captures(none) dereferenceable(65536) %0, ptr noalias nofree readonly align 256 captures(none) dereferenceable(65536) %1, ptr noalias nofree readonly align 256 captures(none) dereferenceable(458752) %2, ptr noalias nofree readonly align 256 captures(none) dereferenceable(112) %3, ptr noalias nofree readonly align 256 captures(none) dereferenceable(1) %4, ptr noalias nofree writeonly align 256 captures(none) dereferenceable(65536) %5) local_unnamed_addr #0 {
  %7 = addrspacecast ptr %4 to ptr addrspace(1)
  %8 = addrspacecast ptr %1 to ptr addrspace(1)
  %9 = addrspacecast ptr %0 to ptr addrspace(1)
  %10 = addrspacecast ptr %3 to ptr addrspace(1)
  %11 = addrspacecast ptr %2 to ptr addrspace(1)
  %12 = addrspacecast ptr %5 to ptr addrspace(1)
  %13 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !4
  %14 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !5
  %15 = load i8, ptr addrspace(1) %7, align 256, !invariant.load !6
  %16 = shl nuw nsw i32 %13, 7
  %17 = or disjoint i32 %16, %14
  %18 = zext nneg i32 %17 to i64
  %19 = getelementptr inbounds [16 x i8], ptr addrspace(1) %8, i64 %18
  %20 = load <2 x double>, ptr addrspace(1) %19, align 16, !invariant.load !6
  %.unpack50 = extractelement <2 x double> %20, i32 0
  %.unpack251 = extractelement <2 x double> %20, i32 1
  %21 = getelementptr inbounds [16 x i8], ptr addrspace(1) %9, i64 %18
  %22 = load <2 x double>, ptr addrspace(1) %21, align 16, !invariant.load !6
  %.unpack352 = extractelement <2 x double> %22, i32 0
  %.unpack553 = extractelement <2 x double> %22, i32 1
  %23 = trunc i8 %15 to i1
  %24 = tail call double @llvm.nvvm.fabs.f64(double %.unpack352)
  %25 = tail call double @llvm.nvvm.fabs.f64(double %.unpack553)
  %26 = tail call double @llvm.maximum.f64(double %24, double %25)
  %27 = tail call double @llvm.minimum.f64(double %24, double %25)
  %28 = fdiv double %27, %26
  %29 = fmul double %28, %28
  %30 = fadd double %29, 1.000000e+00
  %31 = tail call double @llvm.sqrt.f64(double %30)
  %32 = fmul double %26, %31
  %33 = fcmp uno double %32, 0.000000e+00
  %34 = select i1 %33, double %27, double %32
  %35 = select i1 %23, double %.unpack352, double %.unpack50
  %36 = select i1 %23, double %.unpack553, double %.unpack251
  %37 = tail call double @llvm.nvvm.fabs.f64(double %35)
  %38 = tail call double @llvm.nvvm.fabs.f64(double %36)
  %39 = tail call double @llvm.maximum.f64(double %37, double %38)
  %40 = tail call double @llvm.minimum.f64(double %37, double %38)
  %41 = fdiv double %40, %39
  %42 = fmul double %41, %41
  %43 = fadd double %42, 1.000000e+00
  %44 = tail call double @llvm.sqrt.f64(double %43)
  %45 = fmul double %39, %44
  %46 = fcmp uno double %45, 0.000000e+00
  %47 = select i1 %46, double %40, double %45
  %48 = tail call double @llvm.maximum.f64(double %34, double %47)
  %49 = fmul double %48, f0x3EB0C6F7A0B5ED8D
  %50 = load <2 x double>, ptr addrspace(1) %10, align 256, !invariant.load !6
  %.unpack654 = extractelement <2 x double> %50, i32 0
  %.unpack855 = extractelement <2 x double> %50, i32 1
  %51 = getelementptr inbounds [16 x i8], ptr addrspace(1) %11, i64 %18
  %52 = load <2 x double>, ptr addrspace(1) %51, align 16, !invariant.load !6
  %.unpack968 = extractelement <2 x double> %52, i32 0
  %.unpack1169 = extractelement <2 x double> %52, i32 1
  %53 = fmul double %.unpack654, %.unpack968
  %54 = fmul double %.unpack855, %.unpack1169
  %55 = fmul double %.unpack855, %.unpack968
  %56 = fmul double %.unpack654, %.unpack1169
  %57 = fsub double %53, %54
  %58 = fadd double %55, %56
  %59 = fadd double %57, 0.000000e+00
  %60 = fadd double %58, 0.000000e+00
  %61 = getelementptr inbounds i8, ptr addrspace(1) %10, i64 16
  %62 = load <2 x double>, ptr addrspace(1) %61, align 16, !invariant.load !6
  %.unpack1256 = extractelement <2 x double> %62, i32 0
  %.unpack1457 = extractelement <2 x double> %62, i32 1
  %63 = getelementptr inbounds i8, ptr addrspace(1) %51, i64 65536
  %64 = load <2 x double>, ptr addrspace(1) %63, align 16, !invariant.load !6
  %.unpack1570 = extractelement <2 x double> %64, i32 0
  %.unpack1771 = extractelement <2 x double> %64, i32 1
  %65 = fmul double %.unpack1256, %.unpack1570
  %66 = fmul double %.unpack1457, %.unpack1771
  %67 = fmul double %.unpack1457, %.unpack1570
  %68 = fmul double %.unpack1256, %.unpack1771
  %69 = fsub double %65, %66
  %70 = fadd double %67, %68
  %71 = fadd double %59, %69
  %72 = fadd double %60, %70
  %73 = getelementptr inbounds i8, ptr addrspace(1) %10, i64 32
  %74 = load <2 x double>, ptr addrspace(1) %73, align 32, !invariant.load !6
  %.unpack1858 = extractelement <2 x double> %74, i32 0
  %.unpack2059 = extractelement <2 x double> %74, i32 1
  %75 = getelementptr inbounds i8, ptr addrspace(1) %51, i64 131072
  %76 = load <2 x double>, ptr addrspace(1) %75, align 16, !invariant.load !6
  %.unpack2172 = extractelement <2 x double> %76, i32 0
  %.unpack2373 = extractelement <2 x double> %76, i32 1
  %77 = fmul double %.unpack1858, %.unpack2172
  %78 = fmul double %.unpack2059, %.unpack2373
  %79 = fmul double %.unpack2059, %.unpack2172
  %80 = fmul double %.unpack1858, %.unpack2373
  %81 = fsub double %77, %78
  %82 = fadd double %79, %80
  %83 = fadd double %71, %81
  %84 = fadd double %72, %82
  %85 = getelementptr inbounds i8, ptr addrspace(1) %10, i64 48
  %86 = load <2 x double>, ptr addrspace(1) %85, align 16, !invariant.load !6
  %.unpack2460 = extractelement <2 x double> %86, i32 0
  %.unpack2661 = extractelement <2 x double> %86, i32 1
  %87 = getelementptr inbounds i8, ptr addrspace(1) %51, i64 196608
  %88 = load <2 x double>, ptr addrspace(1) %87, align 16, !invariant.load !6
  %.unpack2774 = extractelement <2 x double> %88, i32 0
  %.unpack2975 = extractelement <2 x double> %88, i32 1
  %89 = fmul double %.unpack2460, %.unpack2774
  %90 = fmul double %.unpack2661, %.unpack2975
  %91 = fmul double %.unpack2661, %.unpack2774
  %92 = fmul double %.unpack2460, %.unpack2975
  %93 = fsub double %89, %90
  %94 = fadd double %91, %92
  %95 = fadd double %83, %93
  %96 = fadd double %84, %94
  %97 = getelementptr inbounds i8, ptr addrspace(1) %10, i64 64
  %98 = load <2 x double>, ptr addrspace(1) %97, align 64, !invariant.load !6
  %.unpack3062 = extractelement <2 x double> %98, i32 0
  %.unpack3263 = extractelement <2 x double> %98, i32 1
  %99 = getelementptr inbounds i8, ptr addrspace(1) %51, i64 262144
  %100 = load <2 x double>, ptr addrspace(1) %99, align 16, !invariant.load !6
  %.unpack3376 = extractelement <2 x double> %100, i32 0
  %.unpack3577 = extractelement <2 x double> %100, i32 1
  %101 = fmul double %.unpack3062, %.unpack3376
  %102 = fmul double %.unpack3263, %.unpack3577
  %103 = fmul double %.unpack3263, %.unpack3376
  %104 = fmul double %.unpack3062, %.unpack3577
  %105 = fsub double %101, %102
  %106 = fadd double %103, %104
  %107 = fadd double %95, %105
  %108 = fadd double %96, %106
  %109 = getelementptr inbounds i8, ptr addrspace(1) %10, i64 80
  %110 = load <2 x double>, ptr addrspace(1) %109, align 16, !invariant.load !6
  %.unpack3664 = extractelement <2 x double> %110, i32 0
  %.unpack3865 = extractelement <2 x double> %110, i32 1
  %111 = getelementptr inbounds i8, ptr addrspace(1) %51, i64 327680
  %112 = load <2 x double>, ptr addrspace(1) %111, align 16, !invariant.load !6
  %.unpack3978 = extractelement <2 x double> %112, i32 0
  %.unpack4179 = extractelement <2 x double> %112, i32 1
  %113 = fmul double %.unpack3664, %.unpack3978
  %114 = fmul double %.unpack3865, %.unpack4179
  %115 = fmul double %.unpack3865, %.unpack3978
  %116 = fmul double %.unpack3664, %.unpack4179
  %117 = fsub double %113, %114
  %118 = fadd double %115, %116
  %119 = fadd double %107, %117
  %120 = fadd double %108, %118
  %121 = getelementptr inbounds i8, ptr addrspace(1) %10, i64 96
  %122 = load <2 x double>, ptr addrspace(1) %121, align 32, !invariant.load !6
  %.unpack4266 = extractelement <2 x double> %122, i32 0
  %.unpack4467 = extractelement <2 x double> %122, i32 1
  %123 = getelementptr inbounds i8, ptr addrspace(1) %51, i64 393216
  %124 = load <2 x double>, ptr addrspace(1) %123, align 16, !invariant.load !6
  %.unpack4580 = extractelement <2 x double> %124, i32 0
  %.unpack4781 = extractelement <2 x double> %124, i32 1
  %125 = fmul double %.unpack4266, %.unpack4580
  %126 = fmul double %.unpack4467, %.unpack4781
  %127 = fmul double %.unpack4467, %.unpack4580
  %128 = fmul double %.unpack4266, %.unpack4781
  %129 = fsub double %125, %126
  %130 = fadd double %127, %128
  %131 = fadd double %119, %129
  %132 = fadd double %120, %130
  %133 = fcmp uno double %131, %132
  %134 = fadd double %49, f0x3EB0C6F7A0B5ED8D
  %135 = select i1 %133, double +inf, double %131
  %136 = select i1 %133, double 0.000000e+00, double %132
  %137 = fdiv double %134, 0.000000e+00
  %138 = fmul double %134, %137
  %139 = fmul double %137, %135
  %140 = fadd double %136, %139
  %141 = fdiv double %140, %138
  %142 = fmul double %137, %136
  %143 = fsub double %142, %135
  %144 = fdiv double %143, %138
  %145 = fdiv double 0.000000e+00, %134
  %146 = fmul double %145, 0.000000e+00
  %147 = fadd double %134, %146
  %148 = fmul double %145, %136
  %149 = fadd double %135, %148
  %150 = fdiv double %149, %147
  %151 = fmul double %145, %135
  %152 = fsub double %136, %151
  %153 = fdiv double %152, %147
  %154 = tail call double @llvm.nvvm.fabs.f64(double %134)
  %155 = fcmp oeq double %154, 0.000000e+00
  %156 = fcmp ord double %135, 0.000000e+00
  %157 = fcmp ord double %136, 0.000000e+00
  %158 = or i1 %156, %157
  %159 = and i1 %155, %158
  %160 = tail call double @llvm.copysign.f64(double +inf, double %134)
  %161 = fmul double %160, %135
  %162 = fmul double %160, %136
  %163 = fcmp one double %154, +inf
  %164 = tail call double @llvm.nvvm.fabs.f64(double %135)
  %165 = fcmp oeq double %164, +inf
  %166 = tail call double @llvm.nvvm.fabs.f64(double %136)
  %167 = fcmp oeq double %166, +inf
  %168 = or i1 %165, %167
  %169 = and i1 %163, %168
  %170 = select i1 %165, double 1.000000e+00, double 0.000000e+00
  %171 = tail call double @llvm.copysign.f64(double %170, double %135)
  %172 = select i1 %167, double 1.000000e+00, double 0.000000e+00
  %173 = tail call double @llvm.copysign.f64(double %172, double %136)
  %174 = fmul double %134, %171
  %175 = tail call double @llvm.copysign.f64(double 0.000000e+00, double %136)
  %176 = fadd double %175, %174
  %177 = fmul double %176, +inf
  %178 = tail call double @llvm.copysign.f64(double 0.000000e+00, double %135)
  %179 = fmul double %134, %173
  %180 = fsub double %179, %178
  %181 = fmul double %180, +inf
  %182 = fcmp one double %164, +inf
  %183 = fcmp one double %166, +inf
  %184 = and i1 %182, %183
  %185 = fcmp oeq double %154, +inf
  %186 = and i1 %185, %184
  %187 = select i1 %185, double 1.000000e+00, double 0.000000e+00
  %188 = tail call double @llvm.copysign.f64(double %187, double %134)
  %189 = fmul double %188, %135
  %190 = fmul double %136, 0.000000e+00
  %191 = fadd double %189, %190
  %192 = fmul double %191, 0.000000e+00
  %193 = fmul double %188, %136
  %194 = fmul double %135, 0.000000e+00
  %195 = fsub double %193, %194
  %196 = fmul double %195, 0.000000e+00
  %197 = fcmp olt double %154, 0.000000e+00
  %198 = select i1 %197, double %141, double %150
  %199 = select i1 %197, double %144, double %153
  %200 = select i1 %186, double %192, double %198
  %201 = select i1 %186, double %196, double %199
  %202 = select i1 %169, double %177, double %200
  %203 = select i1 %169, double %181, double %201
  %204 = select i1 %159, double %161, double %202
  %205 = select i1 %159, double %162, double %203
  %206 = fcmp uno double %198, 0.000000e+00
  %207 = fcmp uno double %199, 0.000000e+00
  %208 = and i1 %206, %207
  %209 = select i1 %208, double %204, double %198
  %210 = select i1 %208, double %205, double %199
  %211 = fmul double %209, %209
  %212 = fmul double %209, %210
  %213 = fmul double %210, %210
  %214 = fadd double %211, %213
  %215 = fsub double %212, %212
  %216 = getelementptr inbounds [16 x i8], ptr addrspace(1) %12, i64 %18
  %217 = insertelement <2 x double> poison, double %214, i32 0
  %218 = insertelement <2 x double> %217, double %215, i32 1
  store <2 x double> %218, ptr addrspace(1) %216, align 16
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.maximum.f64(double, double) #2

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.minimum.f64(double, double) #2

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.nvvm.fabs.f64(double) #2

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.copysign.f64(double, double) #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.sqrt.f64(double) #3

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "nvvm.reqntid"="128,1,1" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}
!nvvmir.version = !{!3}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}
!3 = !{i32 2, i32 0}
!4 = !{i32 0, i32 32}
!5 = !{i32 0, i32 128}
!6 = !{}
