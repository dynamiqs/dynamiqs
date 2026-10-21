; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@shared_0 = private addrspace(3) global [8 x { double, double }] undef

declare double @__nv_sqrt(double)

define ptx_kernel void @input_reduce_fusion_2(ptr noalias align 256 dereferenceable(65536) %0, ptr noalias align 256 dereferenceable(8) %1) #0 {
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !1
  %4 = getelementptr inbounds [4096 x { double, double }], ptr %0, i32 0, i32 %3
  %5 = load { double, double }, ptr %4, align 8, !invariant.load !2
  %6 = call { double, double } @scalar_add_computation_add_97_0({ double, double } zeroinitializer, { double, double } %5)
  %7 = add i32 %3, 256
  %8 = getelementptr inbounds [4096 x { double, double }], ptr %0, i32 0, i32 %7
  %9 = load { double, double }, ptr %8, align 8, !invariant.load !2
  %10 = call { double, double } @scalar_add_computation_add_97_0({ double, double } %6, { double, double } %9)
  %11 = add i32 %3, 512
  %12 = getelementptr inbounds [4096 x { double, double }], ptr %0, i32 0, i32 %11
  %13 = load { double, double }, ptr %12, align 8, !invariant.load !2
  %14 = call { double, double } @scalar_add_computation_add_97_0({ double, double } %10, { double, double } %13)
  %15 = add i32 %3, 768
  %16 = getelementptr inbounds [4096 x { double, double }], ptr %0, i32 0, i32 %15
  %17 = load { double, double }, ptr %16, align 8, !invariant.load !2
  %18 = call { double, double } @scalar_add_computation_add_97_0({ double, double } %14, { double, double } %17)
  %19 = add i32 %3, 1024
  %20 = getelementptr inbounds [4096 x { double, double }], ptr %0, i32 0, i32 %19
  %21 = load { double, double }, ptr %20, align 8, !invariant.load !2
  %22 = call { double, double } @scalar_add_computation_add_97_0({ double, double } %18, { double, double } %21)
  %23 = add i32 %3, 1280
  %24 = getelementptr inbounds [4096 x { double, double }], ptr %0, i32 0, i32 %23
  %25 = load { double, double }, ptr %24, align 8, !invariant.load !2
  %26 = call { double, double } @scalar_add_computation_add_97_0({ double, double } %22, { double, double } %25)
  %27 = add i32 %3, 1536
  %28 = getelementptr inbounds [4096 x { double, double }], ptr %0, i32 0, i32 %27
  %29 = load { double, double }, ptr %28, align 8, !invariant.load !2
  %30 = call { double, double } @scalar_add_computation_add_97_0({ double, double } %26, { double, double } %29)
  %31 = add i32 %3, 1792
  %32 = getelementptr inbounds [4096 x { double, double }], ptr %0, i32 0, i32 %31
  %33 = load { double, double }, ptr %32, align 8, !invariant.load !2
  %34 = call { double, double } @scalar_add_computation_add_97_0({ double, double } %30, { double, double } %33)
  %35 = add i32 %3, 2048
  %36 = getelementptr inbounds [4096 x { double, double }], ptr %0, i32 0, i32 %35
  %37 = load { double, double }, ptr %36, align 8, !invariant.load !2
  %38 = call { double, double } @scalar_add_computation_add_97_0({ double, double } %34, { double, double } %37)
  %39 = add i32 %3, 2304
  %40 = getelementptr inbounds [4096 x { double, double }], ptr %0, i32 0, i32 %39
  %41 = load { double, double }, ptr %40, align 8, !invariant.load !2
  %42 = call { double, double } @scalar_add_computation_add_97_0({ double, double } %38, { double, double } %41)
  %43 = add i32 %3, 2560
  %44 = getelementptr inbounds [4096 x { double, double }], ptr %0, i32 0, i32 %43
  %45 = load { double, double }, ptr %44, align 8, !invariant.load !2
  %46 = call { double, double } @scalar_add_computation_add_97_0({ double, double } %42, { double, double } %45)
  %47 = add i32 %3, 2816
  %48 = getelementptr inbounds [4096 x { double, double }], ptr %0, i32 0, i32 %47
  %49 = load { double, double }, ptr %48, align 8, !invariant.load !2
  %50 = call { double, double } @scalar_add_computation_add_97_0({ double, double } %46, { double, double } %49)
  %51 = add i32 %3, 3072
  %52 = getelementptr inbounds [4096 x { double, double }], ptr %0, i32 0, i32 %51
  %53 = load { double, double }, ptr %52, align 8, !invariant.load !2
  %54 = call { double, double } @scalar_add_computation_add_97_0({ double, double } %50, { double, double } %53)
  %55 = add i32 %3, 3328
  %56 = getelementptr inbounds [4096 x { double, double }], ptr %0, i32 0, i32 %55
  %57 = load { double, double }, ptr %56, align 8, !invariant.load !2
  %58 = call { double, double } @scalar_add_computation_add_97_0({ double, double } %54, { double, double } %57)
  %59 = add i32 %3, 3584
  %60 = getelementptr inbounds [4096 x { double, double }], ptr %0, i32 0, i32 %59
  %61 = load { double, double }, ptr %60, align 8, !invariant.load !2
  %62 = call { double, double } @scalar_add_computation_add_97_0({ double, double } %58, { double, double } %61)
  %63 = add i32 %3, 3840
  %64 = getelementptr inbounds [4096 x { double, double }], ptr %0, i32 0, i32 %63
  %65 = load { double, double }, ptr %64, align 8, !invariant.load !2
  %66 = call { double, double } @scalar_add_computation_add_97_0({ double, double } %62, { double, double } %65)
  %67 = extractvalue { double, double } %66, 0
  %68 = bitcast double %67 to i64
  %69 = bitcast i64 %68 to <2 x i32>
  %70 = extractelement <2 x i32> %69, i32 0
  %71 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %70, i32 16, i32 31)
  %72 = insertelement <2 x i32> undef, i32 %71, i32 0
  %73 = extractelement <2 x i32> %69, i32 1
  %74 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %73, i32 16, i32 31)
  %75 = insertelement <2 x i32> %72, i32 %74, i32 1
  %76 = bitcast <2 x i32> %75 to double
  %77 = extractvalue { double, double } %66, 1
  %78 = bitcast double %77 to i64
  %79 = bitcast i64 %78 to <2 x i32>
  %80 = extractelement <2 x i32> %79, i32 0
  %81 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %80, i32 16, i32 31)
  %82 = insertelement <2 x i32> undef, i32 %81, i32 0
  %83 = extractelement <2 x i32> %79, i32 1
  %84 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %83, i32 16, i32 31)
  %85 = insertelement <2 x i32> %82, i32 %84, i32 1
  %86 = bitcast <2 x i32> %85 to double
  %87 = insertvalue { double, double } poison, double %76, 0
  %88 = insertvalue { double, double } %87, double %86, 1
  %89 = call { double, double } @scalar_add_computation_add_97_0({ double, double } %66, { double, double } %88)
  %90 = extractvalue { double, double } %89, 0
  %91 = bitcast double %90 to i64
  %92 = bitcast i64 %91 to <2 x i32>
  %93 = extractelement <2 x i32> %92, i32 0
  %94 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %93, i32 8, i32 31)
  %95 = insertelement <2 x i32> undef, i32 %94, i32 0
  %96 = extractelement <2 x i32> %92, i32 1
  %97 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %96, i32 8, i32 31)
  %98 = insertelement <2 x i32> %95, i32 %97, i32 1
  %99 = bitcast <2 x i32> %98 to double
  %100 = extractvalue { double, double } %89, 1
  %101 = bitcast double %100 to i64
  %102 = bitcast i64 %101 to <2 x i32>
  %103 = extractelement <2 x i32> %102, i32 0
  %104 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %103, i32 8, i32 31)
  %105 = insertelement <2 x i32> undef, i32 %104, i32 0
  %106 = extractelement <2 x i32> %102, i32 1
  %107 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %106, i32 8, i32 31)
  %108 = insertelement <2 x i32> %105, i32 %107, i32 1
  %109 = bitcast <2 x i32> %108 to double
  %110 = insertvalue { double, double } poison, double %99, 0
  %111 = insertvalue { double, double } %110, double %109, 1
  %112 = call { double, double } @scalar_add_computation_add_97_0({ double, double } %89, { double, double } %111)
  %113 = extractvalue { double, double } %112, 0
  %114 = bitcast double %113 to i64
  %115 = bitcast i64 %114 to <2 x i32>
  %116 = extractelement <2 x i32> %115, i32 0
  %117 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %116, i32 4, i32 31)
  %118 = insertelement <2 x i32> undef, i32 %117, i32 0
  %119 = extractelement <2 x i32> %115, i32 1
  %120 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %119, i32 4, i32 31)
  %121 = insertelement <2 x i32> %118, i32 %120, i32 1
  %122 = bitcast <2 x i32> %121 to double
  %123 = extractvalue { double, double } %112, 1
  %124 = bitcast double %123 to i64
  %125 = bitcast i64 %124 to <2 x i32>
  %126 = extractelement <2 x i32> %125, i32 0
  %127 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %126, i32 4, i32 31)
  %128 = insertelement <2 x i32> undef, i32 %127, i32 0
  %129 = extractelement <2 x i32> %125, i32 1
  %130 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %129, i32 4, i32 31)
  %131 = insertelement <2 x i32> %128, i32 %130, i32 1
  %132 = bitcast <2 x i32> %131 to double
  %133 = insertvalue { double, double } poison, double %122, 0
  %134 = insertvalue { double, double } %133, double %132, 1
  %135 = call { double, double } @scalar_add_computation_add_97_0({ double, double } %112, { double, double } %134)
  %136 = extractvalue { double, double } %135, 0
  %137 = bitcast double %136 to i64
  %138 = bitcast i64 %137 to <2 x i32>
  %139 = extractelement <2 x i32> %138, i32 0
  %140 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %139, i32 2, i32 31)
  %141 = insertelement <2 x i32> undef, i32 %140, i32 0
  %142 = extractelement <2 x i32> %138, i32 1
  %143 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %142, i32 2, i32 31)
  %144 = insertelement <2 x i32> %141, i32 %143, i32 1
  %145 = bitcast <2 x i32> %144 to double
  %146 = extractvalue { double, double } %135, 1
  %147 = bitcast double %146 to i64
  %148 = bitcast i64 %147 to <2 x i32>
  %149 = extractelement <2 x i32> %148, i32 0
  %150 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %149, i32 2, i32 31)
  %151 = insertelement <2 x i32> undef, i32 %150, i32 0
  %152 = extractelement <2 x i32> %148, i32 1
  %153 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %152, i32 2, i32 31)
  %154 = insertelement <2 x i32> %151, i32 %153, i32 1
  %155 = bitcast <2 x i32> %154 to double
  %156 = insertvalue { double, double } poison, double %145, 0
  %157 = insertvalue { double, double } %156, double %155, 1
  %158 = call { double, double } @scalar_add_computation_add_97_0({ double, double } %135, { double, double } %157)
  %159 = extractvalue { double, double } %158, 0
  %160 = bitcast double %159 to i64
  %161 = bitcast i64 %160 to <2 x i32>
  %162 = extractelement <2 x i32> %161, i32 0
  %163 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %162, i32 1, i32 31)
  %164 = insertelement <2 x i32> undef, i32 %163, i32 0
  %165 = extractelement <2 x i32> %161, i32 1
  %166 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %165, i32 1, i32 31)
  %167 = insertelement <2 x i32> %164, i32 %166, i32 1
  %168 = bitcast <2 x i32> %167 to double
  %169 = extractvalue { double, double } %158, 1
  %170 = bitcast double %169 to i64
  %171 = bitcast i64 %170 to <2 x i32>
  %172 = extractelement <2 x i32> %171, i32 0
  %173 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %172, i32 1, i32 31)
  %174 = insertelement <2 x i32> undef, i32 %173, i32 0
  %175 = extractelement <2 x i32> %171, i32 1
  %176 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %175, i32 1, i32 31)
  %177 = insertelement <2 x i32> %174, i32 %176, i32 1
  %178 = bitcast <2 x i32> %177 to double
  %179 = insertvalue { double, double } poison, double %168, 0
  %180 = insertvalue { double, double } %179, double %178, 1
  %181 = call { double, double } @scalar_add_computation_add_97_0({ double, double } %158, { double, double } %180)
  %182 = urem i32 %3, 32
  %183 = icmp eq i32 %182, 0
  br i1 %183, label %184, label %187

184:                                              ; preds = %2
  %185 = udiv i32 %3, 32
  %186 = getelementptr inbounds [8 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %185
  store { double, double } %181, ptr %186, align 8
  br label %187

187:                                              ; preds = %184, %2
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %188 = icmp sle i32 %3, 31
  br i1 %188, label %189, label %319

189:                                              ; preds = %187
  %190 = icmp sle i32 %3, 7
  br i1 %190, label %191, label %194

191:                                              ; preds = %189
  %192 = getelementptr inbounds [8 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %3
  %193 = load { double, double }, ptr %192, align 8
  br label %195

194:                                              ; preds = %189
  br label %195

195:                                              ; preds = %191, %194
  %196 = phi { double, double } [ zeroinitializer, %194 ], [ %193, %191 ]
  br label %197

197:                                              ; preds = %195
  %198 = extractvalue { double, double } %196, 0
  %199 = bitcast double %198 to i64
  %200 = bitcast i64 %199 to <2 x i32>
  %201 = extractelement <2 x i32> %200, i32 0
  %202 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %201, i32 16, i32 31)
  %203 = insertelement <2 x i32> undef, i32 %202, i32 0
  %204 = extractelement <2 x i32> %200, i32 1
  %205 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %204, i32 16, i32 31)
  %206 = insertelement <2 x i32> %203, i32 %205, i32 1
  %207 = bitcast <2 x i32> %206 to double
  %208 = extractvalue { double, double } %196, 1
  %209 = bitcast double %208 to i64
  %210 = bitcast i64 %209 to <2 x i32>
  %211 = extractelement <2 x i32> %210, i32 0
  %212 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %211, i32 16, i32 31)
  %213 = insertelement <2 x i32> undef, i32 %212, i32 0
  %214 = extractelement <2 x i32> %210, i32 1
  %215 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %214, i32 16, i32 31)
  %216 = insertelement <2 x i32> %213, i32 %215, i32 1
  %217 = bitcast <2 x i32> %216 to double
  %218 = insertvalue { double, double } poison, double %207, 0
  %219 = insertvalue { double, double } %218, double %217, 1
  %220 = call { double, double } @scalar_add_computation_add_97_0({ double, double } %196, { double, double } %219)
  %221 = extractvalue { double, double } %220, 0
  %222 = bitcast double %221 to i64
  %223 = bitcast i64 %222 to <2 x i32>
  %224 = extractelement <2 x i32> %223, i32 0
  %225 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %224, i32 8, i32 31)
  %226 = insertelement <2 x i32> undef, i32 %225, i32 0
  %227 = extractelement <2 x i32> %223, i32 1
  %228 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %227, i32 8, i32 31)
  %229 = insertelement <2 x i32> %226, i32 %228, i32 1
  %230 = bitcast <2 x i32> %229 to double
  %231 = extractvalue { double, double } %220, 1
  %232 = bitcast double %231 to i64
  %233 = bitcast i64 %232 to <2 x i32>
  %234 = extractelement <2 x i32> %233, i32 0
  %235 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %234, i32 8, i32 31)
  %236 = insertelement <2 x i32> undef, i32 %235, i32 0
  %237 = extractelement <2 x i32> %233, i32 1
  %238 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %237, i32 8, i32 31)
  %239 = insertelement <2 x i32> %236, i32 %238, i32 1
  %240 = bitcast <2 x i32> %239 to double
  %241 = insertvalue { double, double } poison, double %230, 0
  %242 = insertvalue { double, double } %241, double %240, 1
  %243 = call { double, double } @scalar_add_computation_add_97_0({ double, double } %220, { double, double } %242)
  %244 = extractvalue { double, double } %243, 0
  %245 = bitcast double %244 to i64
  %246 = bitcast i64 %245 to <2 x i32>
  %247 = extractelement <2 x i32> %246, i32 0
  %248 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %247, i32 4, i32 31)
  %249 = insertelement <2 x i32> undef, i32 %248, i32 0
  %250 = extractelement <2 x i32> %246, i32 1
  %251 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %250, i32 4, i32 31)
  %252 = insertelement <2 x i32> %249, i32 %251, i32 1
  %253 = bitcast <2 x i32> %252 to double
  %254 = extractvalue { double, double } %243, 1
  %255 = bitcast double %254 to i64
  %256 = bitcast i64 %255 to <2 x i32>
  %257 = extractelement <2 x i32> %256, i32 0
  %258 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %257, i32 4, i32 31)
  %259 = insertelement <2 x i32> undef, i32 %258, i32 0
  %260 = extractelement <2 x i32> %256, i32 1
  %261 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %260, i32 4, i32 31)
  %262 = insertelement <2 x i32> %259, i32 %261, i32 1
  %263 = bitcast <2 x i32> %262 to double
  %264 = insertvalue { double, double } poison, double %253, 0
  %265 = insertvalue { double, double } %264, double %263, 1
  %266 = call { double, double } @scalar_add_computation_add_97_0({ double, double } %243, { double, double } %265)
  %267 = extractvalue { double, double } %266, 0
  %268 = bitcast double %267 to i64
  %269 = bitcast i64 %268 to <2 x i32>
  %270 = extractelement <2 x i32> %269, i32 0
  %271 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %270, i32 2, i32 31)
  %272 = insertelement <2 x i32> undef, i32 %271, i32 0
  %273 = extractelement <2 x i32> %269, i32 1
  %274 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %273, i32 2, i32 31)
  %275 = insertelement <2 x i32> %272, i32 %274, i32 1
  %276 = bitcast <2 x i32> %275 to double
  %277 = extractvalue { double, double } %266, 1
  %278 = bitcast double %277 to i64
  %279 = bitcast i64 %278 to <2 x i32>
  %280 = extractelement <2 x i32> %279, i32 0
  %281 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %280, i32 2, i32 31)
  %282 = insertelement <2 x i32> undef, i32 %281, i32 0
  %283 = extractelement <2 x i32> %279, i32 1
  %284 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %283, i32 2, i32 31)
  %285 = insertelement <2 x i32> %282, i32 %284, i32 1
  %286 = bitcast <2 x i32> %285 to double
  %287 = insertvalue { double, double } poison, double %276, 0
  %288 = insertvalue { double, double } %287, double %286, 1
  %289 = call { double, double } @scalar_add_computation_add_97_0({ double, double } %266, { double, double } %288)
  %290 = extractvalue { double, double } %289, 0
  %291 = bitcast double %290 to i64
  %292 = bitcast i64 %291 to <2 x i32>
  %293 = extractelement <2 x i32> %292, i32 0
  %294 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %293, i32 1, i32 31)
  %295 = insertelement <2 x i32> undef, i32 %294, i32 0
  %296 = extractelement <2 x i32> %292, i32 1
  %297 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %296, i32 1, i32 31)
  %298 = insertelement <2 x i32> %295, i32 %297, i32 1
  %299 = bitcast <2 x i32> %298 to double
  %300 = extractvalue { double, double } %289, 1
  %301 = bitcast double %300 to i64
  %302 = bitcast i64 %301 to <2 x i32>
  %303 = extractelement <2 x i32> %302, i32 0
  %304 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %303, i32 1, i32 31)
  %305 = insertelement <2 x i32> undef, i32 %304, i32 0
  %306 = extractelement <2 x i32> %302, i32 1
  %307 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %306, i32 1, i32 31)
  %308 = insertelement <2 x i32> %305, i32 %307, i32 1
  %309 = bitcast <2 x i32> %308 to double
  %310 = insertvalue { double, double } poison, double %299, 0
  %311 = insertvalue { double, double } %310, double %309, 1
  %312 = call { double, double } @scalar_add_computation_add_97_0({ double, double } %289, { double, double } %311)
  %313 = extractvalue { double, double } %312, 0
  %314 = call double @__nv_sqrt(double %313)
  %315 = icmp eq i32 %3, 0
  br i1 %315, label %316, label %318

316:                                              ; preds = %197
  %317 = getelementptr inbounds [1 x double], ptr %1, i32 0, i32 0
  store double %314, ptr %317, align 8
  br label %318

318:                                              ; preds = %316, %197
  br label %319

319:                                              ; preds = %318, %187
  ret void
}

define internal { double, double } @scalar_add_computation_add_97_0({ double, double } %0, { double, double } %1) {
  %3 = extractvalue { double, double } %0, 0
  %4 = extractvalue { double, double } %1, 0
  %5 = fadd double %3, %4
  %6 = extractvalue { double, double } %0, 1
  %7 = extractvalue { double, double } %1, 1
  %8 = fadd double %6, %7
  %9 = insertvalue { double, double } poison, double %5, 0
  %10 = insertvalue { double, double } %9, double %8, 1
  ret { double, double } %10
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: convergent nocallback nounwind memory(inaccessiblemem: readwrite)
declare i32 @llvm.nvvm.shfl.sync.down.i32(i32, i32, i32, i32) #2

; Function Attrs: convergent nocallback nounwind
declare void @llvm.nvvm.barrier.cta.sync.aligned.all(i32) #3

attributes #0 = { "nvvm.reqntid"="256,1,1" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent nocallback nounwind memory(inaccessiblemem: readwrite) }
attributes #3 = { convergent nocallback nounwind }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 0, i32 256}
!2 = !{}
