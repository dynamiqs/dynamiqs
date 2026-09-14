; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@shared_0 = private unnamed_addr addrspace(3) global [1056 x { double, double }] undef

; Function Attrs: norecurse nounwind
define ptx_kernel void @input_transpose_fusion_3(ptr noalias nofree readonly align 16 captures(none) dereferenceable(65536) %0, ptr noalias nofree writeonly align 256 captures(none) dereferenceable(4096) %1) local_unnamed_addr #0 {
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
  %.unpack95 = extractelement <2 x double> %19, i32 0
  %.unpack296 = extractelement <2 x double> %19, i32 1
  %20 = mul nuw nsw i32 %15, 33
  %21 = add nuw nsw i32 %20, %9
  %22 = zext nneg i32 %21 to i64
  %23 = getelementptr inbounds [16 x i8], ptr addrspace(3) @shared_0, i64 %22
  store double %.unpack95, ptr addrspace(3) %23, align 8
  %.repack3 = getelementptr inbounds i8, ptr addrspace(3) %23, i64 8
  store double %.unpack296, ptr addrspace(3) %.repack3, align 8
  %24 = getelementptr inbounds i8, ptr addrspace(1) %18, i64 4096
  %25 = load <2 x double>, ptr addrspace(1) %24, align 16, !invariant.load !6
  %.unpack597 = extractelement <2 x double> %25, i32 0
  %.unpack798 = extractelement <2 x double> %25, i32 1
  %26 = getelementptr inbounds i8, ptr addrspace(3) %23, i64 64
  store double %.unpack597, ptr addrspace(3) %26, align 8
  %.repack8 = getelementptr inbounds i8, ptr addrspace(3) %23, i64 72
  store double %.unpack798, ptr addrspace(3) %.repack8, align 8
  %27 = getelementptr inbounds i8, ptr addrspace(1) %18, i64 8192
  %28 = load <2 x double>, ptr addrspace(1) %27, align 16, !invariant.load !6
  %.unpack1099 = extractelement <2 x double> %28, i32 0
  %.unpack12100 = extractelement <2 x double> %28, i32 1
  %29 = getelementptr inbounds i8, ptr addrspace(3) %23, i64 128
  store double %.unpack1099, ptr addrspace(3) %29, align 8
  %.repack13 = getelementptr inbounds i8, ptr addrspace(3) %23, i64 136
  store double %.unpack12100, ptr addrspace(3) %.repack13, align 8
  %30 = getelementptr inbounds i8, ptr addrspace(1) %18, i64 12288
  %31 = load <2 x double>, ptr addrspace(1) %30, align 16, !invariant.load !6
  %.unpack15101 = extractelement <2 x double> %31, i32 0
  %.unpack17102 = extractelement <2 x double> %31, i32 1
  %32 = getelementptr inbounds i8, ptr addrspace(3) %23, i64 192
  store double %.unpack15101, ptr addrspace(3) %32, align 8
  %.repack18 = getelementptr inbounds i8, ptr addrspace(3) %23, i64 200
  store double %.unpack17102, ptr addrspace(3) %.repack18, align 8
  %33 = getelementptr inbounds i8, ptr addrspace(1) %18, i64 16384
  %34 = load <2 x double>, ptr addrspace(1) %33, align 16, !invariant.load !6
  %.unpack20103 = extractelement <2 x double> %34, i32 0
  %.unpack22104 = extractelement <2 x double> %34, i32 1
  %35 = getelementptr inbounds i8, ptr addrspace(3) %23, i64 256
  store double %.unpack20103, ptr addrspace(3) %35, align 8
  %.repack23 = getelementptr inbounds i8, ptr addrspace(3) %23, i64 264
  store double %.unpack22104, ptr addrspace(3) %.repack23, align 8
  %36 = getelementptr inbounds i8, ptr addrspace(1) %18, i64 20480
  %37 = load <2 x double>, ptr addrspace(1) %36, align 16, !invariant.load !6
  %.unpack25105 = extractelement <2 x double> %37, i32 0
  %.unpack27106 = extractelement <2 x double> %37, i32 1
  %38 = getelementptr inbounds i8, ptr addrspace(3) %23, i64 320
  store double %.unpack25105, ptr addrspace(3) %38, align 8
  %.repack28 = getelementptr inbounds i8, ptr addrspace(3) %23, i64 328
  store double %.unpack27106, ptr addrspace(3) %.repack28, align 8
  %39 = getelementptr inbounds i8, ptr addrspace(1) %18, i64 24576
  %40 = load <2 x double>, ptr addrspace(1) %39, align 16, !invariant.load !6
  %.unpack30107 = extractelement <2 x double> %40, i32 0
  %.unpack32108 = extractelement <2 x double> %40, i32 1
  %41 = getelementptr inbounds i8, ptr addrspace(3) %23, i64 384
  store double %.unpack30107, ptr addrspace(3) %41, align 8
  %.repack33 = getelementptr inbounds i8, ptr addrspace(3) %23, i64 392
  store double %.unpack32108, ptr addrspace(3) %.repack33, align 8
  %42 = getelementptr inbounds i8, ptr addrspace(1) %18, i64 28672
  %43 = load <2 x double>, ptr addrspace(1) %42, align 16, !invariant.load !6
  %.unpack35109 = extractelement <2 x double> %43, i32 0
  %.unpack37110 = extractelement <2 x double> %43, i32 1
  %44 = getelementptr inbounds i8, ptr addrspace(3) %23, i64 448
  store double %.unpack35109, ptr addrspace(3) %44, align 8
  %.repack38 = getelementptr inbounds i8, ptr addrspace(3) %23, i64 456
  store double %.unpack37110, ptr addrspace(3) %.repack38, align 8
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %45 = mul nuw nsw i32 %9, 33
  %46 = add nuw nsw i32 %45, %15
  %47 = zext nneg i32 %46 to i64
  %48 = getelementptr inbounds [16 x i8], ptr addrspace(3) @shared_0, i64 %47
  %.unpack40 = load double, ptr addrspace(3) %48, align 8
  %.elt41 = getelementptr inbounds i8, ptr addrspace(3) %48, i64 8
  %.unpack42 = load double, ptr addrspace(3) %.elt41, align 8
  %49 = shl nuw nsw i32 %7, 11
  %50 = shl nuw nsw i32 %11, 5
  %51 = or disjoint i32 %49, %50
  %52 = or disjoint i32 %51, %10
  %53 = or disjoint i32 %52, %15
  %54 = zext nneg i32 %53 to i64
  %55 = getelementptr inbounds [16 x i8], ptr addrspace(1) %3, i64 %54
  %56 = load <2 x double>, ptr addrspace(1) %55, align 16, !invariant.load !6
  %.unpack43111 = extractelement <2 x double> %56, i32 0
  %.unpack45112 = extractelement <2 x double> %56, i32 1
  %57 = fneg double %.unpack45112
  %58 = tail call double @llvm.nvvm.fabs.f64(double %.unpack43111)
  %59 = tail call double @llvm.nvvm.fabs.f64(double %57)
  %60 = tail call double @llvm.maximum.f64(double %58, double %59)
  %61 = tail call double @llvm.minimum.f64(double %58, double %59)
  %62 = fdiv double %61, %60
  %63 = fmul double %62, %62
  %64 = fadd double %63, 1.000000e+00
  %65 = tail call double @llvm.sqrt.f64(double %64)
  %66 = fmul double %60, %65
  %67 = fcmp uno double %66, 0.000000e+00
  %68 = select i1 %67, double %61, double %66
  %69 = fsub double %.unpack40, %.unpack43111
  %70 = fadd double %.unpack42, %.unpack45112
  %71 = fmul double %68, 1.000000e-05
  %72 = fcmp one double %58, +inf
  %73 = fcmp one double %59, +inf
  %74 = tail call double @llvm.nvvm.fabs.f64(double %69)
  %75 = tail call double @llvm.nvvm.fabs.f64(double %70)
  %76 = tail call double @llvm.maximum.f64(double %74, double %75)
  %77 = tail call double @llvm.minimum.f64(double %74, double %75)
  %78 = fdiv double %77, %76
  %79 = fmul double %78, %78
  %80 = fadd double %79, 1.000000e+00
  %81 = tail call double @llvm.sqrt.f64(double %80)
  %82 = fmul double %76, %81
  %83 = fcmp uno double %82, 0.000000e+00
  %84 = select i1 %83, double %77, double %82
  %85 = fadd double %71, 1.000000e-05
  %86 = and i1 %72, %73
  %87 = fcmp ole double %84, %85
  %88 = fcmp oeq double %.unpack40, %.unpack43111
  %89 = fcmp oeq double %.unpack42, %57
  %90 = and i1 %88, %89
  %91 = and i1 %86, %87
  %92 = or i1 %90, %91
  %93 = zext i1 %92 to i8
  %94 = getelementptr inbounds i8, ptr addrspace(1) %4, i64 %54
  store i8 %93, ptr addrspace(1) %94, align 1
  %95 = getelementptr inbounds i8, ptr addrspace(3) %48, i64 2112
  %.unpack46 = load double, ptr addrspace(3) %95, align 8
  %.elt47 = getelementptr inbounds i8, ptr addrspace(3) %48, i64 2120
  %.unpack48 = load double, ptr addrspace(3) %.elt47, align 8
  %96 = getelementptr inbounds i8, ptr addrspace(1) %55, i64 4096
  %97 = load <2 x double>, ptr addrspace(1) %96, align 16, !invariant.load !6
  %.unpack49113 = extractelement <2 x double> %97, i32 0
  %.unpack51114 = extractelement <2 x double> %97, i32 1
  %98 = fneg double %.unpack51114
  %99 = tail call double @llvm.nvvm.fabs.f64(double %.unpack49113)
  %100 = tail call double @llvm.nvvm.fabs.f64(double %98)
  %101 = tail call double @llvm.maximum.f64(double %99, double %100)
  %102 = tail call double @llvm.minimum.f64(double %99, double %100)
  %103 = fdiv double %102, %101
  %104 = fmul double %103, %103
  %105 = fadd double %104, 1.000000e+00
  %106 = tail call double @llvm.sqrt.f64(double %105)
  %107 = fmul double %101, %106
  %108 = fcmp uno double %107, 0.000000e+00
  %109 = select i1 %108, double %102, double %107
  %110 = fsub double %.unpack46, %.unpack49113
  %111 = fadd double %.unpack48, %.unpack51114
  %112 = fmul double %109, 1.000000e-05
  %113 = fcmp one double %99, +inf
  %114 = fcmp one double %100, +inf
  %115 = tail call double @llvm.nvvm.fabs.f64(double %110)
  %116 = tail call double @llvm.nvvm.fabs.f64(double %111)
  %117 = tail call double @llvm.maximum.f64(double %115, double %116)
  %118 = tail call double @llvm.minimum.f64(double %115, double %116)
  %119 = fdiv double %118, %117
  %120 = fmul double %119, %119
  %121 = fadd double %120, 1.000000e+00
  %122 = tail call double @llvm.sqrt.f64(double %121)
  %123 = fmul double %117, %122
  %124 = fcmp uno double %123, 0.000000e+00
  %125 = select i1 %124, double %118, double %123
  %126 = fadd double %112, 1.000000e-05
  %127 = and i1 %113, %114
  %128 = fcmp ole double %125, %126
  %129 = fcmp oeq double %.unpack46, %.unpack49113
  %130 = fcmp oeq double %.unpack48, %98
  %131 = and i1 %129, %130
  %132 = and i1 %127, %128
  %133 = or i1 %131, %132
  %134 = zext i1 %133 to i8
  %135 = getelementptr inbounds i8, ptr addrspace(1) %94, i64 256
  store i8 %134, ptr addrspace(1) %135, align 1
  %136 = getelementptr inbounds i8, ptr addrspace(3) %48, i64 4224
  %.unpack52 = load double, ptr addrspace(3) %136, align 8
  %.elt53 = getelementptr inbounds i8, ptr addrspace(3) %48, i64 4232
  %.unpack54 = load double, ptr addrspace(3) %.elt53, align 8
  %137 = getelementptr inbounds i8, ptr addrspace(1) %55, i64 8192
  %138 = load <2 x double>, ptr addrspace(1) %137, align 16, !invariant.load !6
  %.unpack55115 = extractelement <2 x double> %138, i32 0
  %.unpack57116 = extractelement <2 x double> %138, i32 1
  %139 = fneg double %.unpack57116
  %140 = tail call double @llvm.nvvm.fabs.f64(double %.unpack55115)
  %141 = tail call double @llvm.nvvm.fabs.f64(double %139)
  %142 = tail call double @llvm.maximum.f64(double %140, double %141)
  %143 = tail call double @llvm.minimum.f64(double %140, double %141)
  %144 = fdiv double %143, %142
  %145 = fmul double %144, %144
  %146 = fadd double %145, 1.000000e+00
  %147 = tail call double @llvm.sqrt.f64(double %146)
  %148 = fmul double %142, %147
  %149 = fcmp uno double %148, 0.000000e+00
  %150 = select i1 %149, double %143, double %148
  %151 = fsub double %.unpack52, %.unpack55115
  %152 = fadd double %.unpack54, %.unpack57116
  %153 = fmul double %150, 1.000000e-05
  %154 = fcmp one double %140, +inf
  %155 = fcmp one double %141, +inf
  %156 = tail call double @llvm.nvvm.fabs.f64(double %151)
  %157 = tail call double @llvm.nvvm.fabs.f64(double %152)
  %158 = tail call double @llvm.maximum.f64(double %156, double %157)
  %159 = tail call double @llvm.minimum.f64(double %156, double %157)
  %160 = fdiv double %159, %158
  %161 = fmul double %160, %160
  %162 = fadd double %161, 1.000000e+00
  %163 = tail call double @llvm.sqrt.f64(double %162)
  %164 = fmul double %158, %163
  %165 = fcmp uno double %164, 0.000000e+00
  %166 = select i1 %165, double %159, double %164
  %167 = fadd double %153, 1.000000e-05
  %168 = and i1 %154, %155
  %169 = fcmp ole double %166, %167
  %170 = fcmp oeq double %.unpack52, %.unpack55115
  %171 = fcmp oeq double %.unpack54, %139
  %172 = and i1 %170, %171
  %173 = and i1 %168, %169
  %174 = or i1 %172, %173
  %175 = zext i1 %174 to i8
  %176 = getelementptr inbounds i8, ptr addrspace(1) %94, i64 512
  store i8 %175, ptr addrspace(1) %176, align 1
  %177 = getelementptr inbounds i8, ptr addrspace(3) %48, i64 6336
  %.unpack58 = load double, ptr addrspace(3) %177, align 8
  %.elt59 = getelementptr inbounds i8, ptr addrspace(3) %48, i64 6344
  %.unpack60 = load double, ptr addrspace(3) %.elt59, align 8
  %178 = getelementptr inbounds i8, ptr addrspace(1) %55, i64 12288
  %179 = load <2 x double>, ptr addrspace(1) %178, align 16, !invariant.load !6
  %.unpack61117 = extractelement <2 x double> %179, i32 0
  %.unpack63118 = extractelement <2 x double> %179, i32 1
  %180 = fneg double %.unpack63118
  %181 = tail call double @llvm.nvvm.fabs.f64(double %.unpack61117)
  %182 = tail call double @llvm.nvvm.fabs.f64(double %180)
  %183 = tail call double @llvm.maximum.f64(double %181, double %182)
  %184 = tail call double @llvm.minimum.f64(double %181, double %182)
  %185 = fdiv double %184, %183
  %186 = fmul double %185, %185
  %187 = fadd double %186, 1.000000e+00
  %188 = tail call double @llvm.sqrt.f64(double %187)
  %189 = fmul double %183, %188
  %190 = fcmp uno double %189, 0.000000e+00
  %191 = select i1 %190, double %184, double %189
  %192 = fsub double %.unpack58, %.unpack61117
  %193 = fadd double %.unpack60, %.unpack63118
  %194 = fmul double %191, 1.000000e-05
  %195 = fcmp one double %181, +inf
  %196 = fcmp one double %182, +inf
  %197 = tail call double @llvm.nvvm.fabs.f64(double %192)
  %198 = tail call double @llvm.nvvm.fabs.f64(double %193)
  %199 = tail call double @llvm.maximum.f64(double %197, double %198)
  %200 = tail call double @llvm.minimum.f64(double %197, double %198)
  %201 = fdiv double %200, %199
  %202 = fmul double %201, %201
  %203 = fadd double %202, 1.000000e+00
  %204 = tail call double @llvm.sqrt.f64(double %203)
  %205 = fmul double %199, %204
  %206 = fcmp uno double %205, 0.000000e+00
  %207 = select i1 %206, double %200, double %205
  %208 = fadd double %194, 1.000000e-05
  %209 = and i1 %195, %196
  %210 = fcmp ole double %207, %208
  %211 = fcmp oeq double %.unpack58, %.unpack61117
  %212 = fcmp oeq double %.unpack60, %180
  %213 = and i1 %211, %212
  %214 = and i1 %209, %210
  %215 = or i1 %213, %214
  %216 = zext i1 %215 to i8
  %217 = getelementptr inbounds i8, ptr addrspace(1) %94, i64 768
  store i8 %216, ptr addrspace(1) %217, align 1
  %218 = getelementptr inbounds i8, ptr addrspace(3) %48, i64 8448
  %.unpack64 = load double, ptr addrspace(3) %218, align 8
  %.elt65 = getelementptr inbounds i8, ptr addrspace(3) %48, i64 8456
  %.unpack66 = load double, ptr addrspace(3) %.elt65, align 8
  %219 = getelementptr inbounds i8, ptr addrspace(1) %55, i64 16384
  %220 = load <2 x double>, ptr addrspace(1) %219, align 16, !invariant.load !6
  %.unpack67119 = extractelement <2 x double> %220, i32 0
  %.unpack69120 = extractelement <2 x double> %220, i32 1
  %221 = fneg double %.unpack69120
  %222 = tail call double @llvm.nvvm.fabs.f64(double %.unpack67119)
  %223 = tail call double @llvm.nvvm.fabs.f64(double %221)
  %224 = tail call double @llvm.maximum.f64(double %222, double %223)
  %225 = tail call double @llvm.minimum.f64(double %222, double %223)
  %226 = fdiv double %225, %224
  %227 = fmul double %226, %226
  %228 = fadd double %227, 1.000000e+00
  %229 = tail call double @llvm.sqrt.f64(double %228)
  %230 = fmul double %224, %229
  %231 = fcmp uno double %230, 0.000000e+00
  %232 = select i1 %231, double %225, double %230
  %233 = fsub double %.unpack64, %.unpack67119
  %234 = fadd double %.unpack66, %.unpack69120
  %235 = fmul double %232, 1.000000e-05
  %236 = fcmp one double %222, +inf
  %237 = fcmp one double %223, +inf
  %238 = tail call double @llvm.nvvm.fabs.f64(double %233)
  %239 = tail call double @llvm.nvvm.fabs.f64(double %234)
  %240 = tail call double @llvm.maximum.f64(double %238, double %239)
  %241 = tail call double @llvm.minimum.f64(double %238, double %239)
  %242 = fdiv double %241, %240
  %243 = fmul double %242, %242
  %244 = fadd double %243, 1.000000e+00
  %245 = tail call double @llvm.sqrt.f64(double %244)
  %246 = fmul double %240, %245
  %247 = fcmp uno double %246, 0.000000e+00
  %248 = select i1 %247, double %241, double %246
  %249 = fadd double %235, 1.000000e-05
  %250 = and i1 %236, %237
  %251 = fcmp ole double %248, %249
  %252 = fcmp oeq double %.unpack64, %.unpack67119
  %253 = fcmp oeq double %.unpack66, %221
  %254 = and i1 %252, %253
  %255 = and i1 %250, %251
  %256 = or i1 %254, %255
  %257 = zext i1 %256 to i8
  %258 = getelementptr inbounds i8, ptr addrspace(1) %94, i64 1024
  store i8 %257, ptr addrspace(1) %258, align 1
  %259 = getelementptr inbounds i8, ptr addrspace(3) %48, i64 10560
  %.unpack70 = load double, ptr addrspace(3) %259, align 8
  %.elt71 = getelementptr inbounds i8, ptr addrspace(3) %48, i64 10568
  %.unpack72 = load double, ptr addrspace(3) %.elt71, align 8
  %260 = getelementptr inbounds i8, ptr addrspace(1) %55, i64 20480
  %261 = load <2 x double>, ptr addrspace(1) %260, align 16, !invariant.load !6
  %.unpack73121 = extractelement <2 x double> %261, i32 0
  %.unpack75122 = extractelement <2 x double> %261, i32 1
  %262 = fneg double %.unpack75122
  %263 = tail call double @llvm.nvvm.fabs.f64(double %.unpack73121)
  %264 = tail call double @llvm.nvvm.fabs.f64(double %262)
  %265 = tail call double @llvm.maximum.f64(double %263, double %264)
  %266 = tail call double @llvm.minimum.f64(double %263, double %264)
  %267 = fdiv double %266, %265
  %268 = fmul double %267, %267
  %269 = fadd double %268, 1.000000e+00
  %270 = tail call double @llvm.sqrt.f64(double %269)
  %271 = fmul double %265, %270
  %272 = fcmp uno double %271, 0.000000e+00
  %273 = select i1 %272, double %266, double %271
  %274 = fsub double %.unpack70, %.unpack73121
  %275 = fadd double %.unpack72, %.unpack75122
  %276 = fmul double %273, 1.000000e-05
  %277 = fcmp one double %263, +inf
  %278 = fcmp one double %264, +inf
  %279 = tail call double @llvm.nvvm.fabs.f64(double %274)
  %280 = tail call double @llvm.nvvm.fabs.f64(double %275)
  %281 = tail call double @llvm.maximum.f64(double %279, double %280)
  %282 = tail call double @llvm.minimum.f64(double %279, double %280)
  %283 = fdiv double %282, %281
  %284 = fmul double %283, %283
  %285 = fadd double %284, 1.000000e+00
  %286 = tail call double @llvm.sqrt.f64(double %285)
  %287 = fmul double %281, %286
  %288 = fcmp uno double %287, 0.000000e+00
  %289 = select i1 %288, double %282, double %287
  %290 = fadd double %276, 1.000000e-05
  %291 = and i1 %277, %278
  %292 = fcmp ole double %289, %290
  %293 = fcmp oeq double %.unpack70, %.unpack73121
  %294 = fcmp oeq double %.unpack72, %262
  %295 = and i1 %293, %294
  %296 = and i1 %291, %292
  %297 = or i1 %295, %296
  %298 = zext i1 %297 to i8
  %299 = getelementptr inbounds i8, ptr addrspace(1) %94, i64 1280
  store i8 %298, ptr addrspace(1) %299, align 1
  %300 = getelementptr inbounds i8, ptr addrspace(3) %48, i64 12672
  %.unpack76 = load double, ptr addrspace(3) %300, align 8
  %.elt77 = getelementptr inbounds i8, ptr addrspace(3) %48, i64 12680
  %.unpack78 = load double, ptr addrspace(3) %.elt77, align 8
  %301 = getelementptr inbounds i8, ptr addrspace(1) %55, i64 24576
  %302 = load <2 x double>, ptr addrspace(1) %301, align 16, !invariant.load !6
  %.unpack79123 = extractelement <2 x double> %302, i32 0
  %.unpack81124 = extractelement <2 x double> %302, i32 1
  %303 = fneg double %.unpack81124
  %304 = tail call double @llvm.nvvm.fabs.f64(double %.unpack79123)
  %305 = tail call double @llvm.nvvm.fabs.f64(double %303)
  %306 = tail call double @llvm.maximum.f64(double %304, double %305)
  %307 = tail call double @llvm.minimum.f64(double %304, double %305)
  %308 = fdiv double %307, %306
  %309 = fmul double %308, %308
  %310 = fadd double %309, 1.000000e+00
  %311 = tail call double @llvm.sqrt.f64(double %310)
  %312 = fmul double %306, %311
  %313 = fcmp uno double %312, 0.000000e+00
  %314 = select i1 %313, double %307, double %312
  %315 = fsub double %.unpack76, %.unpack79123
  %316 = fadd double %.unpack78, %.unpack81124
  %317 = fmul double %314, 1.000000e-05
  %318 = fcmp one double %304, +inf
  %319 = fcmp one double %305, +inf
  %320 = tail call double @llvm.nvvm.fabs.f64(double %315)
  %321 = tail call double @llvm.nvvm.fabs.f64(double %316)
  %322 = tail call double @llvm.maximum.f64(double %320, double %321)
  %323 = tail call double @llvm.minimum.f64(double %320, double %321)
  %324 = fdiv double %323, %322
  %325 = fmul double %324, %324
  %326 = fadd double %325, 1.000000e+00
  %327 = tail call double @llvm.sqrt.f64(double %326)
  %328 = fmul double %322, %327
  %329 = fcmp uno double %328, 0.000000e+00
  %330 = select i1 %329, double %323, double %328
  %331 = fadd double %317, 1.000000e-05
  %332 = and i1 %318, %319
  %333 = fcmp ole double %330, %331
  %334 = fcmp oeq double %.unpack76, %.unpack79123
  %335 = fcmp oeq double %.unpack78, %303
  %336 = and i1 %334, %335
  %337 = and i1 %332, %333
  %338 = or i1 %336, %337
  %339 = zext i1 %338 to i8
  %340 = getelementptr inbounds i8, ptr addrspace(1) %94, i64 1536
  store i8 %339, ptr addrspace(1) %340, align 1
  %341 = getelementptr inbounds i8, ptr addrspace(3) %48, i64 14784
  %.unpack82 = load double, ptr addrspace(3) %341, align 8
  %.elt83 = getelementptr inbounds i8, ptr addrspace(3) %48, i64 14792
  %.unpack84 = load double, ptr addrspace(3) %.elt83, align 8
  %342 = getelementptr inbounds i8, ptr addrspace(1) %55, i64 28672
  %343 = load <2 x double>, ptr addrspace(1) %342, align 16, !invariant.load !6
  %.unpack85125 = extractelement <2 x double> %343, i32 0
  %.unpack87126 = extractelement <2 x double> %343, i32 1
  %344 = fneg double %.unpack87126
  %345 = tail call double @llvm.nvvm.fabs.f64(double %.unpack85125)
  %346 = tail call double @llvm.nvvm.fabs.f64(double %344)
  %347 = tail call double @llvm.maximum.f64(double %345, double %346)
  %348 = tail call double @llvm.minimum.f64(double %345, double %346)
  %349 = fdiv double %348, %347
  %350 = fmul double %349, %349
  %351 = fadd double %350, 1.000000e+00
  %352 = tail call double @llvm.sqrt.f64(double %351)
  %353 = fmul double %347, %352
  %354 = fcmp uno double %353, 0.000000e+00
  %355 = select i1 %354, double %348, double %353
  %356 = fsub double %.unpack82, %.unpack85125
  %357 = fadd double %.unpack84, %.unpack87126
  %358 = fmul double %355, 1.000000e-05
  %359 = fcmp one double %345, +inf
  %360 = fcmp one double %346, +inf
  %361 = tail call double @llvm.nvvm.fabs.f64(double %356)
  %362 = tail call double @llvm.nvvm.fabs.f64(double %357)
  %363 = tail call double @llvm.maximum.f64(double %361, double %362)
  %364 = tail call double @llvm.minimum.f64(double %361, double %362)
  %365 = fdiv double %364, %363
  %366 = fmul double %365, %365
  %367 = fadd double %366, 1.000000e+00
  %368 = tail call double @llvm.sqrt.f64(double %367)
  %369 = fmul double %363, %368
  %370 = fcmp uno double %369, 0.000000e+00
  %371 = select i1 %370, double %364, double %369
  %372 = fadd double %358, 1.000000e-05
  %373 = and i1 %359, %360
  %374 = fcmp ole double %371, %372
  %375 = fcmp oeq double %.unpack82, %.unpack85125
  %376 = fcmp oeq double %.unpack84, %344
  %377 = and i1 %375, %376
  %378 = and i1 %373, %374
  %379 = or i1 %377, %378
  %380 = zext i1 %379 to i8
  %381 = getelementptr inbounds i8, ptr addrspace(1) %94, i64 1792
  store i8 %380, ptr addrspace(1) %381, align 1
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: convergent nocallback nounwind
declare void @llvm.nvvm.barrier.cta.sync.aligned.all(i32) #2

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.maximum.f64(double, double) #3

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.minimum.f64(double, double) #3

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.nvvm.fabs.f64(double) #3

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.sqrt.f64(double) #4

attributes #0 = { norecurse nounwind "nvvm.reqntid"="128,1,1" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent nocallback nounwind }
attributes #3 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
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
