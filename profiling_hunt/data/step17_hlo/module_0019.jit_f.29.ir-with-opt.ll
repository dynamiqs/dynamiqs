; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@shared_0 = private unnamed_addr addrspace(3) global [8 x { double, double }] undef

; Function Attrs: norecurse nounwind
define ptx_kernel void @input_reduce_fusion_1(ptr noalias nofree readonly align 256 captures(none) dereferenceable(65536) %0, ptr noalias nofree writeonly align 256 captures(none) dereferenceable(8) %1) local_unnamed_addr #0 {
  %3 = addrspacecast ptr %0 to ptr addrspace(1)
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !4
  %6 = zext nneg i32 %5 to i64
  %7 = getelementptr inbounds [16 x i8], ptr addrspace(1) %3, i64 %6
  %8 = load <2 x double>, ptr addrspace(1) %7, align 16, !invariant.load !5
  %.unpack54 = extractelement <2 x double> %8, i32 0
  %.unpack255 = extractelement <2 x double> %8, i32 1
  %9 = fadd double %.unpack54, 0.000000e+00
  %10 = fadd double %.unpack255, 0.000000e+00
  %11 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 4096
  %12 = load <2 x double>, ptr addrspace(1) %11, align 16, !invariant.load !5
  %.unpack356 = extractelement <2 x double> %12, i32 0
  %.unpack557 = extractelement <2 x double> %12, i32 1
  %13 = fadd double %9, %.unpack356
  %14 = fadd double %10, %.unpack557
  %15 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 8192
  %16 = load <2 x double>, ptr addrspace(1) %15, align 16, !invariant.load !5
  %.unpack658 = extractelement <2 x double> %16, i32 0
  %.unpack859 = extractelement <2 x double> %16, i32 1
  %17 = fadd double %13, %.unpack658
  %18 = fadd double %14, %.unpack859
  %19 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 12288
  %20 = load <2 x double>, ptr addrspace(1) %19, align 16, !invariant.load !5
  %.unpack960 = extractelement <2 x double> %20, i32 0
  %.unpack1161 = extractelement <2 x double> %20, i32 1
  %21 = fadd double %17, %.unpack960
  %22 = fadd double %18, %.unpack1161
  %23 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 16384
  %24 = load <2 x double>, ptr addrspace(1) %23, align 16, !invariant.load !5
  %.unpack1262 = extractelement <2 x double> %24, i32 0
  %.unpack1463 = extractelement <2 x double> %24, i32 1
  %25 = fadd double %21, %.unpack1262
  %26 = fadd double %22, %.unpack1463
  %27 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 20480
  %28 = load <2 x double>, ptr addrspace(1) %27, align 16, !invariant.load !5
  %.unpack1564 = extractelement <2 x double> %28, i32 0
  %.unpack1765 = extractelement <2 x double> %28, i32 1
  %29 = fadd double %25, %.unpack1564
  %30 = fadd double %26, %.unpack1765
  %31 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 24576
  %32 = load <2 x double>, ptr addrspace(1) %31, align 16, !invariant.load !5
  %.unpack1866 = extractelement <2 x double> %32, i32 0
  %.unpack2067 = extractelement <2 x double> %32, i32 1
  %33 = fadd double %29, %.unpack1866
  %34 = fadd double %30, %.unpack2067
  %35 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 28672
  %36 = load <2 x double>, ptr addrspace(1) %35, align 16, !invariant.load !5
  %.unpack2168 = extractelement <2 x double> %36, i32 0
  %.unpack2369 = extractelement <2 x double> %36, i32 1
  %37 = fadd double %33, %.unpack2168
  %38 = fadd double %34, %.unpack2369
  %39 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 32768
  %40 = load <2 x double>, ptr addrspace(1) %39, align 16, !invariant.load !5
  %.unpack2470 = extractelement <2 x double> %40, i32 0
  %.unpack2671 = extractelement <2 x double> %40, i32 1
  %41 = fadd double %37, %.unpack2470
  %42 = fadd double %38, %.unpack2671
  %43 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 36864
  %44 = load <2 x double>, ptr addrspace(1) %43, align 16, !invariant.load !5
  %.unpack2772 = extractelement <2 x double> %44, i32 0
  %.unpack2973 = extractelement <2 x double> %44, i32 1
  %45 = fadd double %41, %.unpack2772
  %46 = fadd double %42, %.unpack2973
  %47 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 40960
  %48 = load <2 x double>, ptr addrspace(1) %47, align 16, !invariant.load !5
  %.unpack3074 = extractelement <2 x double> %48, i32 0
  %.unpack3275 = extractelement <2 x double> %48, i32 1
  %49 = fadd double %45, %.unpack3074
  %50 = fadd double %46, %.unpack3275
  %51 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 45056
  %52 = load <2 x double>, ptr addrspace(1) %51, align 16, !invariant.load !5
  %.unpack3376 = extractelement <2 x double> %52, i32 0
  %.unpack3577 = extractelement <2 x double> %52, i32 1
  %53 = fadd double %49, %.unpack3376
  %54 = fadd double %50, %.unpack3577
  %55 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 49152
  %56 = load <2 x double>, ptr addrspace(1) %55, align 16, !invariant.load !5
  %.unpack3678 = extractelement <2 x double> %56, i32 0
  %.unpack3879 = extractelement <2 x double> %56, i32 1
  %57 = fadd double %53, %.unpack3678
  %58 = fadd double %54, %.unpack3879
  %59 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 53248
  %60 = load <2 x double>, ptr addrspace(1) %59, align 16, !invariant.load !5
  %.unpack3980 = extractelement <2 x double> %60, i32 0
  %.unpack4181 = extractelement <2 x double> %60, i32 1
  %61 = fadd double %57, %.unpack3980
  %62 = fadd double %58, %.unpack4181
  %63 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 57344
  %64 = load <2 x double>, ptr addrspace(1) %63, align 16, !invariant.load !5
  %.unpack4282 = extractelement <2 x double> %64, i32 0
  %.unpack4483 = extractelement <2 x double> %64, i32 1
  %65 = fadd double %61, %.unpack4282
  %66 = fadd double %62, %.unpack4483
  %67 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 61440
  %68 = load <2 x double>, ptr addrspace(1) %67, align 16, !invariant.load !5
  %.unpack4584 = extractelement <2 x double> %68, i32 0
  %.unpack4785 = extractelement <2 x double> %68, i32 1
  %69 = fadd double %65, %.unpack4584
  %70 = fadd double %66, %.unpack4785
  %71 = bitcast double %69 to <2 x i32>
  %72 = extractelement <2 x i32> %71, i64 0
  %73 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %72, i32 16, i32 31)
  %74 = insertelement <2 x i32> poison, i32 %73, i64 0
  %75 = extractelement <2 x i32> %71, i64 1
  %76 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %75, i32 16, i32 31)
  %77 = insertelement <2 x i32> %74, i32 %76, i64 1
  %78 = bitcast <2 x i32> %77 to double
  %79 = bitcast double %70 to <2 x i32>
  %80 = extractelement <2 x i32> %79, i64 0
  %81 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %80, i32 16, i32 31)
  %82 = insertelement <2 x i32> poison, i32 %81, i64 0
  %83 = extractelement <2 x i32> %79, i64 1
  %84 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %83, i32 16, i32 31)
  %85 = insertelement <2 x i32> %82, i32 %84, i64 1
  %86 = bitcast <2 x i32> %85 to double
  %87 = fadd double %69, %78
  %88 = fadd double %70, %86
  %89 = bitcast double %87 to <2 x i32>
  %90 = extractelement <2 x i32> %89, i64 0
  %91 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %90, i32 8, i32 31)
  %92 = insertelement <2 x i32> poison, i32 %91, i64 0
  %93 = extractelement <2 x i32> %89, i64 1
  %94 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %93, i32 8, i32 31)
  %95 = insertelement <2 x i32> %92, i32 %94, i64 1
  %96 = bitcast <2 x i32> %95 to double
  %97 = bitcast double %88 to <2 x i32>
  %98 = extractelement <2 x i32> %97, i64 0
  %99 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %98, i32 8, i32 31)
  %100 = insertelement <2 x i32> poison, i32 %99, i64 0
  %101 = extractelement <2 x i32> %97, i64 1
  %102 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %101, i32 8, i32 31)
  %103 = insertelement <2 x i32> %100, i32 %102, i64 1
  %104 = bitcast <2 x i32> %103 to double
  %105 = fadd double %87, %96
  %106 = fadd double %88, %104
  %107 = bitcast double %105 to <2 x i32>
  %108 = extractelement <2 x i32> %107, i64 0
  %109 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %108, i32 4, i32 31)
  %110 = insertelement <2 x i32> poison, i32 %109, i64 0
  %111 = extractelement <2 x i32> %107, i64 1
  %112 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %111, i32 4, i32 31)
  %113 = insertelement <2 x i32> %110, i32 %112, i64 1
  %114 = bitcast <2 x i32> %113 to double
  %115 = bitcast double %106 to <2 x i32>
  %116 = extractelement <2 x i32> %115, i64 0
  %117 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %116, i32 4, i32 31)
  %118 = insertelement <2 x i32> poison, i32 %117, i64 0
  %119 = extractelement <2 x i32> %115, i64 1
  %120 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %119, i32 4, i32 31)
  %121 = insertelement <2 x i32> %118, i32 %120, i64 1
  %122 = bitcast <2 x i32> %121 to double
  %123 = fadd double %105, %114
  %124 = fadd double %106, %122
  %125 = bitcast double %123 to <2 x i32>
  %126 = extractelement <2 x i32> %125, i64 0
  %127 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %126, i32 2, i32 31)
  %128 = insertelement <2 x i32> poison, i32 %127, i64 0
  %129 = extractelement <2 x i32> %125, i64 1
  %130 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %129, i32 2, i32 31)
  %131 = insertelement <2 x i32> %128, i32 %130, i64 1
  %132 = bitcast <2 x i32> %131 to double
  %133 = bitcast double %124 to <2 x i32>
  %134 = extractelement <2 x i32> %133, i64 0
  %135 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %134, i32 2, i32 31)
  %136 = insertelement <2 x i32> poison, i32 %135, i64 0
  %137 = extractelement <2 x i32> %133, i64 1
  %138 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %137, i32 2, i32 31)
  %139 = insertelement <2 x i32> %136, i32 %138, i64 1
  %140 = bitcast <2 x i32> %139 to double
  %141 = fadd double %123, %132
  %142 = fadd double %124, %140
  %143 = bitcast double %141 to <2 x i32>
  %144 = extractelement <2 x i32> %143, i64 0
  %145 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %144, i32 1, i32 31)
  %146 = extractelement <2 x i32> %143, i64 1
  %147 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %146, i32 1, i32 31)
  %148 = bitcast double %142 to <2 x i32>
  %149 = extractelement <2 x i32> %148, i64 0
  %150 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %149, i32 1, i32 31)
  %151 = extractelement <2 x i32> %148, i64 1
  %152 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %151, i32 1, i32 31)
  %153 = and i32 %5, 31
  %154 = icmp eq i32 %153, 0
  br i1 %154, label %155, label %168

155:                                              ; preds = %2
  %156 = trunc i64 %6 to i32
  %157 = lshr exact i32 %156, 5
  %158 = zext nneg i32 %157 to i64
  %159 = getelementptr inbounds [16 x i8], ptr addrspace(3) @shared_0, i64 %158
  %.repack48 = getelementptr inbounds i8, ptr addrspace(3) %159, i64 8
  %160 = insertelement <2 x i32> poison, i32 %150, i64 0
  %161 = insertelement <2 x i32> %160, i32 %152, i64 1
  %162 = bitcast <2 x i32> %161 to double
  %163 = fadd double %142, %162
  %164 = insertelement <2 x i32> poison, i32 %145, i64 0
  %165 = insertelement <2 x i32> %164, i32 %147, i64 1
  %166 = bitcast <2 x i32> %165 to double
  %167 = fadd double %141, %166
  store double %167, ptr addrspace(3) %159, align 8
  store double %163, ptr addrspace(3) %.repack48, align 8
  br label %168

168:                                              ; preds = %155, %2
  %169 = trunc i64 %6 to i32
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %170 = icmp samesign ult i32 %169, 32
  br i1 %170, label %171, label %272

171:                                              ; preds = %168
  %172 = trunc i64 %6 to i32
  %173 = icmp samesign ult i32 %172, 8
  %174 = getelementptr inbounds [16 x i8], ptr addrspace(3) @shared_0, i64 %6
  br i1 %173, label %175, label %178

175:                                              ; preds = %171
  %.unpack51 = load double, ptr addrspace(3) %174, align 8
  %176 = insertvalue { double, double } poison, double %.unpack51, 0
  %sunkaddr = getelementptr inbounds i8, ptr addrspace(3) %174, i64 8
  %.unpack53 = load double, ptr addrspace(3) %sunkaddr, align 8
  %177 = insertvalue { double, double } %176, double %.unpack53, 1
  br label %178

178:                                              ; preds = %175, %171
  %179 = phi { double, double } [ %177, %175 ], [ zeroinitializer, %171 ]
  %180 = trunc i64 %6 to i32
  %181 = extractvalue { double, double } %179, 0
  %182 = bitcast double %181 to <2 x i32>
  %183 = extractelement <2 x i32> %182, i64 0
  %184 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %183, i32 16, i32 31)
  %185 = insertelement <2 x i32> poison, i32 %184, i64 0
  %186 = extractelement <2 x i32> %182, i64 1
  %187 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %186, i32 16, i32 31)
  %188 = insertelement <2 x i32> %185, i32 %187, i64 1
  %189 = bitcast <2 x i32> %188 to double
  %190 = extractvalue { double, double } %179, 1
  %191 = bitcast double %190 to <2 x i32>
  %192 = extractelement <2 x i32> %191, i64 0
  %193 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %192, i32 16, i32 31)
  %194 = insertelement <2 x i32> poison, i32 %193, i64 0
  %195 = extractelement <2 x i32> %191, i64 1
  %196 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %195, i32 16, i32 31)
  %197 = insertelement <2 x i32> %194, i32 %196, i64 1
  %198 = bitcast <2 x i32> %197 to double
  %199 = fadd double %181, %189
  %200 = fadd double %190, %198
  %201 = bitcast double %199 to <2 x i32>
  %202 = extractelement <2 x i32> %201, i64 0
  %203 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %202, i32 8, i32 31)
  %204 = insertelement <2 x i32> poison, i32 %203, i64 0
  %205 = extractelement <2 x i32> %201, i64 1
  %206 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %205, i32 8, i32 31)
  %207 = insertelement <2 x i32> %204, i32 %206, i64 1
  %208 = bitcast <2 x i32> %207 to double
  %209 = bitcast double %200 to <2 x i32>
  %210 = extractelement <2 x i32> %209, i64 0
  %211 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %210, i32 8, i32 31)
  %212 = insertelement <2 x i32> poison, i32 %211, i64 0
  %213 = extractelement <2 x i32> %209, i64 1
  %214 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %213, i32 8, i32 31)
  %215 = insertelement <2 x i32> %212, i32 %214, i64 1
  %216 = bitcast <2 x i32> %215 to double
  %217 = fadd double %199, %208
  %218 = fadd double %200, %216
  %219 = bitcast double %217 to <2 x i32>
  %220 = extractelement <2 x i32> %219, i64 0
  %221 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %220, i32 4, i32 31)
  %222 = insertelement <2 x i32> poison, i32 %221, i64 0
  %223 = extractelement <2 x i32> %219, i64 1
  %224 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %223, i32 4, i32 31)
  %225 = insertelement <2 x i32> %222, i32 %224, i64 1
  %226 = bitcast <2 x i32> %225 to double
  %227 = bitcast double %218 to <2 x i32>
  %228 = extractelement <2 x i32> %227, i64 0
  %229 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %228, i32 4, i32 31)
  %230 = insertelement <2 x i32> poison, i32 %229, i64 0
  %231 = extractelement <2 x i32> %227, i64 1
  %232 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %231, i32 4, i32 31)
  %233 = insertelement <2 x i32> %230, i32 %232, i64 1
  %234 = bitcast <2 x i32> %233 to double
  %235 = fadd double %217, %226
  %236 = fadd double %218, %234
  %237 = bitcast double %235 to <2 x i32>
  %238 = extractelement <2 x i32> %237, i64 0
  %239 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %238, i32 2, i32 31)
  %240 = insertelement <2 x i32> poison, i32 %239, i64 0
  %241 = extractelement <2 x i32> %237, i64 1
  %242 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %241, i32 2, i32 31)
  %243 = insertelement <2 x i32> %240, i32 %242, i64 1
  %244 = bitcast <2 x i32> %243 to double
  %245 = bitcast double %236 to <2 x i32>
  %246 = extractelement <2 x i32> %245, i64 0
  %247 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %246, i32 2, i32 31)
  %248 = insertelement <2 x i32> poison, i32 %247, i64 0
  %249 = extractelement <2 x i32> %245, i64 1
  %250 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %249, i32 2, i32 31)
  %251 = insertelement <2 x i32> %248, i32 %250, i64 1
  %252 = bitcast <2 x i32> %251 to double
  %253 = fadd double %235, %244
  %254 = fadd double %236, %252
  %255 = bitcast double %253 to <2 x i32>
  %256 = extractelement <2 x i32> %255, i64 0
  %257 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %256, i32 1, i32 31)
  %258 = extractelement <2 x i32> %255, i64 1
  %259 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %258, i32 1, i32 31)
  %260 = bitcast double %254 to <2 x i32>
  %261 = extractelement <2 x i32> %260, i64 0
  %262 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %261, i32 1, i32 31)
  %263 = extractelement <2 x i32> %260, i64 1
  %264 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %263, i32 1, i32 31)
  %265 = icmp eq i32 %180, 0
  %266 = insertelement <2 x i32> poison, i32 %257, i64 0
  %267 = insertelement <2 x i32> %266, i32 %259, i64 1
  %268 = bitcast <2 x i32> %267 to double
  %269 = fadd double %253, %268
  %270 = tail call double @llvm.sqrt.f64(double %269)
  br i1 %265, label %271, label %272

271:                                              ; preds = %178
  store double %270, ptr addrspace(1) %4, align 256
  br label %272

272:                                              ; preds = %178, %271, %168
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: convergent nocallback nounwind memory(inaccessiblemem: readwrite)
declare i32 @llvm.nvvm.shfl.sync.down.i32(i32, i32, i32, i32) #2

; Function Attrs: convergent nocallback nounwind
declare void @llvm.nvvm.barrier.cta.sync.aligned.all(i32) #3

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.sqrt.f64(double) #4

attributes #0 = { norecurse nounwind "nvvm.reqntid"="256,1,1" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent nocallback nounwind memory(inaccessiblemem: readwrite) }
attributes #3 = { convergent nocallback nounwind }
attributes #4 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}
!nvvmir.version = !{!3}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}
!3 = !{i32 2, i32 0}
!4 = !{i32 0, i32 256}
!5 = !{}
