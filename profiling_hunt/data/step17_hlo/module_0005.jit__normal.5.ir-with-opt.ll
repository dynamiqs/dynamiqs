; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_maximum_multiply_fusion(ptr noalias nofree readonly align 256 captures(none) dereferenceable(4) %0, ptr noalias nofree readonly align 256 captures(none) dereferenceable(8) %1, ptr noalias nofree readonly align 256 captures(none) dereferenceable(4) %2, ptr noalias nofree writeonly align 256 captures(none) dereferenceable(32768) %3, ptr noalias nofree writeonly align 256 captures(none) dereferenceable(32768) %4) local_unnamed_addr #0 {
  %6 = addrspacecast ptr %2 to ptr addrspace(1)
  %7 = addrspacecast ptr %1 to ptr addrspace(1)
  %8 = addrspacecast ptr %0 to ptr addrspace(1)
  %9 = addrspacecast ptr %3 to ptr addrspace(1)
  %10 = addrspacecast ptr %4 to ptr addrspace(1)
  %11 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !4
  %12 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !5
  %13 = shl nuw nsw i32 %11, 7
  %14 = or disjoint i32 %13, %12
  %15 = load i32, ptr addrspace(1) %6, align 256, !invariant.load !6
  %16 = add i32 %15, %14
  %17 = load i32, ptr addrspace(1) %7, align 256, !invariant.load !6
  %18 = tail call i32 @llvm.fshl.i32(i32 %16, i32 %16, i32 13)
  %19 = add i32 %17, %16
  %20 = xor i32 %19, %18
  %21 = tail call i32 @llvm.fshl.i32(i32 %20, i32 %20, i32 15)
  %22 = add i32 %20, %19
  %23 = xor i32 %22, %21
  %24 = add i32 %23, %22
  %25 = tail call i32 @llvm.fshl.i32(i32 %23, i32 %23, i32 26)
  %26 = xor i32 %24, %25
  %27 = tail call i32 @llvm.fshl.i32(i32 %26, i32 %26, i32 6)
  %28 = add i32 %26, %24
  %29 = xor i32 %28, %27
  %30 = xor i32 %15, %17
  %31 = xor i32 %30, 466688986
  %32 = add i32 %31, 1
  %33 = add i32 %32, %29
  %34 = add i32 %28, %15
  %35 = tail call i32 @llvm.fshl.i32(i32 %33, i32 %33, i32 17)
  %36 = add i32 %34, %33
  %37 = xor i32 %36, %35
  %38 = tail call i32 @llvm.fshl.i32(i32 %37, i32 %37, i32 29)
  %39 = add i32 %37, %36
  %40 = xor i32 %39, %38
  %41 = add i32 %40, %39
  %42 = tail call i32 @llvm.fshl.i32(i32 %40, i32 %40, i32 16)
  %43 = xor i32 %41, %42
  %44 = tail call i32 @llvm.fshl.i32(i32 %43, i32 %43, i32 24)
  %45 = add i32 %43, %41
  %46 = xor i32 %45, %44
  %47 = add i32 %17, 2
  %48 = add i32 %47, %46
  %49 = add i32 %45, %31
  %50 = tail call i32 @llvm.fshl.i32(i32 %48, i32 %48, i32 13)
  %51 = add i32 %49, %48
  %52 = xor i32 %51, %50
  %53 = tail call i32 @llvm.fshl.i32(i32 %52, i32 %52, i32 15)
  %54 = add i32 %52, %51
  %55 = xor i32 %54, %53
  %56 = add i32 %55, %54
  %57 = tail call i32 @llvm.fshl.i32(i32 %55, i32 %55, i32 26)
  %58 = xor i32 %56, %57
  %59 = tail call i32 @llvm.fshl.i32(i32 %58, i32 %58, i32 6)
  %60 = add i32 %58, %56
  %61 = xor i32 %60, %59
  %62 = add i32 %15, 3
  %63 = add i32 %62, %61
  %64 = add i32 %60, %17
  %65 = tail call i32 @llvm.fshl.i32(i32 %63, i32 %63, i32 17)
  %66 = add i32 %64, %63
  %67 = xor i32 %66, %65
  %68 = tail call i32 @llvm.fshl.i32(i32 %67, i32 %67, i32 29)
  %69 = add i32 %67, %66
  %70 = xor i32 %69, %68
  %71 = add i32 %70, %69
  %72 = tail call i32 @llvm.fshl.i32(i32 %70, i32 %70, i32 16)
  %73 = xor i32 %71, %72
  %74 = tail call i32 @llvm.fshl.i32(i32 %73, i32 %73, i32 24)
  %75 = add i32 %73, %71
  %76 = xor i32 %75, %74
  %77 = add i32 %31, 4
  %78 = add i32 %77, %76
  %79 = add i32 %75, %15
  %80 = tail call i32 @llvm.fshl.i32(i32 %78, i32 %78, i32 13)
  %81 = add i32 %79, %78
  %82 = xor i32 %81, %80
  %83 = tail call i32 @llvm.fshl.i32(i32 %82, i32 %82, i32 15)
  %84 = add i32 %82, %81
  %85 = xor i32 %84, %83
  %86 = tail call i32 @llvm.fshl.i32(i32 %85, i32 %85, i32 26)
  %87 = add i32 %85, %84
  %88 = xor i32 %87, %86
  %89 = tail call i32 @llvm.fshl.i32(i32 %88, i32 %88, i32 6)
  %90 = add i32 %88, %87
  %91 = xor i32 %90, %89
  %92 = add i32 %90, %31
  %93 = zext i32 %92 to i64
  %94 = add i32 %17, 5
  %95 = add i32 %94, %91
  %96 = shl nuw i64 %93, 32
  %97 = zext i32 %95 to i64
  %98 = or disjoint i64 %96, %97
  %99 = lshr i64 %98, 12
  %100 = or disjoint i64 %99, 4607182418800017408
  %101 = bitcast i64 %100 to double
  %102 = fadd nnan double %101, -1.000000e+00
  %103 = fmul nnan double %102, 2.000000e+00
  %104 = fadd double %103, f0xBFEFFFFFFFFFFFFF
  %105 = tail call double @llvm.maximum.f64(double %104, double f0xBFEFFFFFFFFFFFFF)
  %106 = fneg double %105
  %107 = fmul double %105, %106
  %108 = tail call i32 @llvm.nvvm.d2i.hi(double %107) #5
  %109 = bitcast i32 %108 to float
  %110 = fcmp olt float %109, f0x3FE55555
  %111 = fcmp ogt float %109, f0xBFD99999
  %or.cond.i = and i1 %110, %111
  br i1 %or.cond.i, label %112, label %129

112:                                              ; preds = %5
  %113 = fadd double %107, 2.000000e+00
  %114 = fdiv double %107, %113
  %115 = fneg double %107
  %116 = fmul double %114, %115
  %117 = fadd double %107, %116
  %118 = fmul double %117, %117
  %119 = tail call double @llvm.fma.f64(double %118, double f0x3EB372FB2FBE14B5, double f0x3ED087FFCEB2DC44)
  %120 = tail call double @llvm.fma.f64(double %119, double %118, double f0x3EF3B9FF890F468C)
  %121 = tail call double @llvm.fma.f64(double %120, double %118, double f0x3F17457EFD51BAF8)
  %122 = tail call double @llvm.fma.f64(double %121, double %118, double f0x3F3C71C8DE3CE825)
  %123 = tail call double @llvm.fma.f64(double %122, double %118, double f0x3F6249248FA4661F)
  %124 = tail call double @llvm.fma.f64(double %123, double %118, double f0x3F899999999D70C4)
  %125 = tail call double @llvm.fma.f64(double %124, double %118, double f0x3FB5555555555462)
  %126 = fmul double %118, %125
  %127 = tail call double @llvm.fma.f64(double %126, double %117, double %116)
  %128 = fadd double %107, %127
  br label %__nv_log1p.exit

129:                                              ; preds = %5
  %130 = fadd double %107, 1.000000e+00
  %131 = tail call i32 @llvm.nvvm.d2i.hi(double %130) #5
  %132 = tail call i32 @llvm.nvvm.d2i.lo(double %130) #5
  %133 = icmp slt i32 %131, 1048576
  %134 = fmul double %130, f0x4350000000000000
  %135 = tail call i32 @llvm.nvvm.d2i.lo(double %134) #5
  %136 = tail call i32 @llvm.nvvm.d2i.hi(double %134) #5
  br i1 %133, label %137, label %138

137:                                              ; preds = %129
  br label %138

138:                                              ; preds = %137, %129
  %.0.i = phi double [ %134, %137 ], [ %130, %129 ]
  %ihi.i.0.i = phi i32 [ %136, %137 ], [ %131, %129 ]
  %ilo.i.0.i = phi i32 [ %135, %137 ], [ %132, %129 ]
  %e.i.0.i = phi i32 [ -1077, %137 ], [ -1023, %129 ]
  %139 = add i32 %ihi.i.0.i, -1
  %or.cond2.i = icmp ult i32 %139, 2146435071
  br i1 %or.cond2.i, label %140, label %187

140:                                              ; preds = %138
  %141 = lshr i32 %ihi.i.0.i, 20
  %142 = add nsw i32 %e.i.0.i, %141
  %143 = and i32 %ihi.i.0.i, 1048575
  %144 = or disjoint i32 %143, 1072693248
  %145 = tail call double @llvm.nvvm.lohi.i2d(i32 %ilo.i.0.i, i32 %144) #5
  %146 = icmp samesign ugt i32 %144, 1073127582
  %147 = tail call i32 @llvm.nvvm.d2i.lo(double %145) #5
  %148 = tail call i32 @llvm.nvvm.d2i.hi(double %145) #5
  %149 = add i32 %148, -1048576
  %150 = tail call double @llvm.nvvm.lohi.i2d(i32 %147, i32 %149) #5
  %spec.select = select i1 %146, double %150, double %145
  %151 = zext i1 %146 to i32
  %spec.select1 = add nsw i32 %142, %151
  %152 = fadd double %spec.select, -1.000000e+00
  %153 = fadd double %spec.select, 1.000000e+00
  %154 = tail call double @llvm.nvvm.rcp.approx.ftz.d(double %153) #5
  %155 = fneg double %153
  %156 = tail call double @llvm.fma.f64(double %155, double %154, double 1.000000e+00)
  %157 = tail call double @llvm.fma.f64(double %156, double %156, double %156)
  %158 = tail call double @llvm.fma.f64(double %157, double %154, double %154)
  %159 = fmul double %152, %158
  %160 = fadd double %159, %159
  %161 = fmul double %160, %160
  %162 = tail call double @llvm.fma.f64(double %161, double f0x3EB1380B3AE80F1E, double f0x3ED0EE258B7A8B04)
  %163 = tail call double @llvm.fma.f64(double %162, double %161, double f0x3EF3B2669F02676F)
  %164 = tail call double @llvm.fma.f64(double %163, double %161, double f0x3F1745CBA9AB0956)
  %165 = tail call double @llvm.fma.f64(double %164, double %161, double f0x3F3C71C72D1B5154)
  %166 = tail call double @llvm.fma.f64(double %165, double %161, double f0x3F624924923BE72D)
  %167 = tail call double @llvm.fma.f64(double %166, double %161, double f0x3F8999999999A3C4)
  %168 = tail call double @llvm.fma.f64(double %167, double %161, double f0x3FB5555555555554)
  %169 = fsub double %152, %160
  %170 = fmul double %169, 2.000000e+00
  %171 = fneg double %160
  %172 = tail call double @llvm.fma.f64(double %171, double %152, double %170)
  %173 = fmul double %158, %172
  %174 = fmul double %161, %168
  %175 = tail call double @llvm.fma.f64(double %174, double %160, double %173)
  %176 = xor i32 %spec.select1, -2147483648
  %177 = tail call double @llvm.nvvm.lohi.i2d(i32 %176, i32 1127219200) #5
  %178 = tail call double @llvm.nvvm.lohi.i2d(i32 -2147483648, i32 1127219200) #5
  %179 = fsub double %177, %178
  %180 = tail call double @llvm.fma.f64(double %179, double f0x3FE62E42FEFA39EF, double %160)
  %181 = fneg double %179
  %182 = tail call double @llvm.fma.f64(double %181, double f0x3FE62E42FEFA39EF, double %180)
  %183 = fsub double %182, %160
  %184 = fsub double %175, %183
  %185 = tail call double @llvm.fma.f64(double %179, double f0x3C7ABC9E3B39803F, double %184)
  %186 = fadd double %180, %185
  br label %__nv_log1p.exit

187:                                              ; preds = %138
  %188 = tail call double @llvm.fma.f64(double %.0.i, double +inf, double +inf)
  %189 = tail call i32 @llvm.nvvm.d2i.hi(double %.0.i) #5
  %190 = and i32 %189, 2147483647
  %191 = icmp eq i32 %190, 0
  %q.i.0.i = select i1 %191, double -inf, double %188
  br label %__nv_log1p.exit

__nv_log1p.exit:                                  ; preds = %112, %140, %187
  %t.0.i = phi double [ %128, %112 ], [ %186, %140 ], [ %q.i.0.i, %187 ]
  %192 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 4
  %193 = load i32, ptr addrspace(1) %192, align 4, !invariant.load !6
  %194 = load i32, ptr addrspace(1) %8, align 256, !invariant.load !6
  %195 = add i32 %194, %14
  %196 = add i32 %195, %193
  %197 = tail call i32 @llvm.fshl.i32(i32 %195, i32 %195, i32 13)
  %198 = xor i32 %196, %197
  %199 = add i32 %198, %196
  %200 = tail call i32 @llvm.fshl.i32(i32 %198, i32 %198, i32 15)
  %201 = xor i32 %199, %200
  %202 = add i32 %201, %199
  %203 = tail call i32 @llvm.fshl.i32(i32 %201, i32 %201, i32 26)
  %204 = xor i32 %202, %203
  %205 = add i32 %204, %202
  %206 = add i32 %205, %194
  %207 = tail call i32 @llvm.fshl.i32(i32 %204, i32 %204, i32 6)
  %208 = xor i32 %205, %207
  %209 = xor i32 %193, %194
  %210 = xor i32 %209, 466688986
  %211 = add i32 %210, 1
  %212 = add i32 %211, %208
  %213 = add i32 %206, %212
  %214 = tail call i32 @llvm.fshl.i32(i32 %212, i32 %212, i32 17)
  %215 = xor i32 %213, %214
  %216 = add i32 %215, %213
  %217 = tail call i32 @llvm.fshl.i32(i32 %215, i32 %215, i32 29)
  %218 = xor i32 %216, %217
  %219 = add i32 %218, %216
  %220 = tail call i32 @llvm.fshl.i32(i32 %218, i32 %218, i32 16)
  %221 = xor i32 %219, %220
  %222 = add i32 %221, %219
  %223 = add i32 %222, %210
  %224 = tail call i32 @llvm.fshl.i32(i32 %221, i32 %221, i32 24)
  %225 = xor i32 %222, %224
  %226 = add i32 %193, 2
  %227 = add i32 %226, %225
  %228 = add i32 %223, %227
  %229 = tail call i32 @llvm.fshl.i32(i32 %227, i32 %227, i32 13)
  %230 = xor i32 %228, %229
  %231 = add i32 %230, %228
  %232 = tail call i32 @llvm.fshl.i32(i32 %230, i32 %230, i32 15)
  %233 = xor i32 %231, %232
  %234 = add i32 %233, %231
  %235 = tail call i32 @llvm.fshl.i32(i32 %233, i32 %233, i32 26)
  %236 = xor i32 %234, %235
  %237 = add i32 %236, %234
  %238 = add i32 %237, %193
  %239 = tail call i32 @llvm.fshl.i32(i32 %236, i32 %236, i32 6)
  %240 = xor i32 %237, %239
  %241 = add i32 %194, 3
  %242 = add i32 %241, %240
  %243 = add i32 %238, %242
  %244 = tail call i32 @llvm.fshl.i32(i32 %242, i32 %242, i32 17)
  %245 = xor i32 %243, %244
  %246 = add i32 %245, %243
  %247 = tail call i32 @llvm.fshl.i32(i32 %245, i32 %245, i32 29)
  %248 = xor i32 %246, %247
  %249 = add i32 %248, %246
  %250 = tail call i32 @llvm.fshl.i32(i32 %248, i32 %248, i32 16)
  %251 = xor i32 %249, %250
  %252 = add i32 %251, %249
  %253 = add i32 %252, %194
  %254 = tail call i32 @llvm.fshl.i32(i32 %251, i32 %251, i32 24)
  %255 = xor i32 %252, %254
  %256 = add i32 %210, 4
  %257 = add i32 %256, %255
  %258 = add i32 %253, %257
  %259 = tail call i32 @llvm.fshl.i32(i32 %257, i32 %257, i32 13)
  %260 = xor i32 %258, %259
  %261 = add i32 %260, %258
  %262 = tail call i32 @llvm.fshl.i32(i32 %260, i32 %260, i32 15)
  %263 = xor i32 %261, %262
  %264 = add i32 %263, %261
  %265 = tail call i32 @llvm.fshl.i32(i32 %263, i32 %263, i32 26)
  %266 = xor i32 %264, %265
  %267 = add i32 %266, %264
  %268 = add i32 %267, %210
  %269 = zext i32 %268 to i64
  %270 = shl nuw i64 %269, 32
  %271 = tail call i32 @llvm.fshl.i32(i32 %266, i32 %266, i32 6)
  %272 = xor i32 %267, %271
  %273 = add i32 %193, 5
  %274 = add i32 %273, %272
  %275 = zext i32 %274 to i64
  %276 = or disjoint i64 %270, %275
  %277 = lshr i64 %276, 12
  %278 = or disjoint i64 %277, 4607182418800017408
  %279 = bitcast i64 %278 to double
  %280 = fadd nnan double %279, -1.000000e+00
  %281 = fmul nnan double %280, 2.000000e+00
  %282 = fadd double %281, f0xBFEFFFFFFFFFFFFF
  %283 = tail call double @llvm.maximum.f64(double %282, double f0xBFEFFFFFFFFFFFFF)
  %284 = fneg double %t.0.i
  %285 = fcmp ogt double %t.0.i, -6.250000e+00
  %286 = select i1 %285, double f0xBBB135D2E746E627, double f0x3E23040F87DBD932
  %287 = select i1 %285, double f0xBC08DDF93324D327, double f0x3E785CBE52878635
  %288 = fcmp ogt double %t.0.i, -1.600000e+01
  %289 = select i1 %288, double %286, double f0xBDBDCEC3A7785389
  %290 = select i1 %288, double %287, double f0xBDF18FEEC0E38727
  %291 = tail call double @llvm.sqrt.f64(double %284)
  %292 = select i1 %288, double 3.250000e+00, double 5.000000e+00
  %293 = fsub double -3.125000e+00, %t.0.i
  %294 = fsub double %291, %292
  %295 = select i1 %285, double %293, double %294
  %296 = fmul double %289, %295
  %297 = select i1 %285, double f0x3C37B83EEF0B7C9F, double f0xBE92777453DD3955
  %298 = fadd double %290, %296
  %299 = select i1 %288, double %297, double f0x3E19E6BF2DDA45E3
  %300 = fmul double %295, %298
  %301 = select i1 %285, double f0x3C69BA72CD589B91, double f0x3E5395ABCD554C6C
  %302 = fadd double %299, %300
  %303 = select i1 %288, double %301, double f0xBE30468FB24E2F5F
  %304 = fmul double %295, %302
  %305 = select i1 %285, double f0xBCA33689090A6B96, double f0x3EB936388A3790AD
  %306 = fadd double %303, %304
  %307 = select i1 %288, double %305, double f0x3E405AC6A8FBA182
  %308 = fmul double %295, %306
  %309 = select i1 %285, double f0x3C782E11898132E0, double f0xBED0D5DB812B5083
  %310 = fadd double %307, %308
  %311 = select i1 %288, double %309, double f0xBE50102E495FB9C0
  %312 = fmul double %295, %310
  %313 = select i1 %285, double f0x3CFDE4ACFD9E26BA, double f0x3EC8860CD5D652F6
  %314 = fadd double %311, %312
  %315 = select i1 %288, double %313, double f0x3E5F4C20E1334AF8
  %316 = fmul double %295, %314
  %317 = select i1 %285, double f0xBD26D33EED66C487, double f0x3EEA29A0CACDFB23
  %318 = fadd double %315, %316
  %319 = select i1 %288, double %317, double f0xBE722D220FDF9C3E
  %320 = fmul double %295, %318
  %321 = select i1 %285, double f0xBD36F2167040D8E2, double f0xBF08CEF1F80281F2
  %322 = fadd double %319, %320
  %323 = select i1 %288, double %321, double f0x3E8EBC8BB824CB54
  %324 = fmul double %295, %322
  %325 = select i1 %285, double f0x3D872A22C2D77E20, double f0x3F11E684D0B9188A
  %326 = fadd double %323, %324
  %327 = select i1 %288, double %325, double f0xBEB0A8D40EA372CC
  %328 = fmul double %295, %326
  %329 = select i1 %285, double f0xBDAC8859C4E5C0AF, double f0x3EF932CD54C8A222
  %330 = fadd double %327, %328
  %331 = select i1 %288, double %329, double f0x3ED2FBD29D093D2B
  %332 = fmul double %295, %330
  %333 = select i1 %285, double f0xBDCDC583D118A561, double f0xBF37448A89EF8AA3
  %334 = fadd double %331, %332
  %335 = select i1 %288, double %333, double f0xBEF4A3497E1E0FAC
  %336 = fmul double %295, %334
  %337 = select i1 %285, double f0x3E120F47CCF46B3C, double f0x3F4F3CC55AD40C25
  %338 = fadd double %335, %336
  %339 = select i1 %288, double %337, double f0x3F13EBF4EB00938F
  %340 = fmul double %295, %338
  %341 = select i1 %285, double f0xBE31A9E38DC84D60, double f0xBF5BA924132F38B1
  %342 = fadd double %339, %340
  %343 = select i1 %288, double %341, double f0xBF2C2F36A8FC5D53
  %344 = fmul double %295, %342
  %345 = select i1 %285, double f0xBE5F36CD6D3D46A9, double f0x3F6468EECA533CF8
  %346 = fadd double %343, %344
  %347 = select i1 %288, double %345, double f0xBF222EA5DF04047C
  %348 = fmul double %295, %346
  %349 = select i1 %285, double f0x3E9C6B4F5D03B787, double f0xBF6EBADABB891BBD
  %350 = fadd double %347, %348
  %351 = select i1 %288, double %349, double f0x3FF02A30D1FBA0DC
  %352 = fmul double %295, %350
  %353 = select i1 %285, double f0xBEB6E8A5434AE8A2, double f0x3F75FFCFE5B76AFC
  %354 = fadd double %351, %352
  %355 = select i1 %285, double f0xBEED1D1F7B8736F6, double f0x3FF0158A6D641D39
  %356 = select i1 %288, double %353, double f0x4013664DDD1AD7FB
  %357 = fmul double %295, %354
  %358 = fadd double %356, %357
  %359 = fmul double %295, %358
  %360 = select i1 %285, double f0x3F2879C2A212F024, double f0x4008ABCC380D5A48
  %361 = fadd double %355, %359
  %362 = select i1 %288, double %361, double %358
  %363 = fmul double %295, %362
  %364 = fadd double %360, %363
  %365 = select i1 %288, double %364, double %358
  %366 = fmul double %295, %365
  %367 = fadd double %366, f0xBF4845769484FCA8
  %368 = select i1 %285, double %367, double %365
  %369 = fmul double %295, %368
  %370 = fadd double %369, f0xBF78B6C33114F909
  %371 = select i1 %285, double %370, double %365
  %372 = fmul double %295, %371
  %373 = fadd double %372, f0x3FCEBD80D9B13E28
  %374 = select i1 %285, double %373, double %365
  %375 = fmul double %295, %374
  %376 = fadd double %375, f0x3FFA755E7C99AE86
  %377 = tail call double @llvm.nvvm.fabs.f64(double %105)
  %378 = select i1 %285, double %376, double %365
  %379 = fcmp oeq double %377, 1.000000e+00
  %.v = select i1 %379, double +inf, double %378
  %380 = fmul double %105, %.v
  %381 = fmul double %380, f0x3FF6A09E667F3BCD
  %382 = zext nneg i32 %14 to i64
  %383 = getelementptr inbounds [8 x i8], ptr addrspace(1) %9, i64 %382
  store double %283, ptr addrspace(1) %383, align 8
  %384 = getelementptr inbounds [8 x i8], ptr addrspace(1) %10, i64 %382
  store double %381, ptr addrspace(1) %384, align 8
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.maximum.f64(double, double) #2

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.nvvm.fabs.f64(double) #2

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.nvvm.d2i.hi(double) #2

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.nvvm.d2i.lo(double) #2

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.nvvm.lohi.i2d(i32, i32) #2

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare double @llvm.nvvm.rcp.approx.ftz.d(double) #3

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.fshl.i32(i32, i32, i32) #4

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.sqrt.f64(double) #4

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fma.f64(double, double, double) #4

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "nvvm.reqntid"="128,1,1" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #4 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #5 = { nounwind }

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
