; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define ptx_kernel void @input_add_fusion(ptr noalias nofree readonly align 256 captures(none) dereferenceable(65536) %0, ptr noalias nofree readonly align 256 captures(none) dereferenceable(65536) %1, ptr noalias nofree readonly align 256 captures(none) dereferenceable(1008) %2, ptr noalias nofree readonly align 256 captures(none) dereferenceable(1008) %3, ptr noalias nofree readonly align 256 captures(none) dereferenceable(992) %4, ptr noalias nofree readonly align 256 captures(none) dereferenceable(1) %5, ptr noalias nofree readonly align 256 captures(none) dereferenceable(992) %6, ptr noalias nofree readonly align 256 captures(none) dereferenceable(512) %7, ptr noalias nofree readonly align 256 captures(none) dereferenceable(1536) %8, ptr noalias nofree writeonly align 256 captures(none) dereferenceable(65536) %9, ptr noalias nofree writeonly align 256 captures(none) dereferenceable(65536) %10) local_unnamed_addr #0 {
  %12 = addrspacecast ptr %0 to ptr addrspace(1)
  %13 = addrspacecast ptr %1 to ptr addrspace(1)
  %14 = addrspacecast ptr %6 to ptr addrspace(1)
  %15 = addrspacecast ptr %4 to ptr addrspace(1)
  %16 = addrspacecast ptr %5 to ptr addrspace(1)
  %17 = addrspacecast ptr %3 to ptr addrspace(1)
  %18 = addrspacecast ptr %2 to ptr addrspace(1)
  %19 = addrspacecast ptr %8 to ptr addrspace(1)
  %20 = addrspacecast ptr %7 to ptr addrspace(1)
  %21 = addrspacecast ptr %9 to ptr addrspace(1)
  %22 = addrspacecast ptr %10 to ptr addrspace(1)
  %23 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !2
  %24 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !3
  %25 = shl nuw nsw i32 %23, 1
  %26 = lshr i32 %24, 6
  %27 = or disjoint i32 %25, %26
  %28 = and i32 %24, 63
  tail call void @llvm.experimental.noalias.scope.decl(metadata !4)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !7)
  %29 = shl nuw nsw i32 %27, 6
  %30 = or disjoint i32 %29, %28
  %31 = zext nneg i32 %30 to i64
  %32 = getelementptr inbounds [16 x i8], ptr addrspace(1) %12, i64 %31
  %33 = load <2 x double>, ptr addrspace(1) %32, align 16, !invariant.load !9, !alias.scope !4, !noalias !7
  %.unpack.i231 = extractelement <2 x double> %33, i32 0
  %.unpack2.i232 = extractelement <2 x double> %33, i32 1
  %34 = shl nuw nsw i32 %28, 6
  %35 = or disjoint i32 %34, %27
  %36 = zext nneg i32 %35 to i64
  %37 = getelementptr inbounds [16 x i8], ptr addrspace(1) %13, i64 %36
  %38 = load <2 x double>, ptr addrspace(1) %37, align 16, !invariant.load !9, !alias.scope !7, !noalias !4
  %.unpack3.i233 = extractelement <2 x double> %38, i32 0
  %.unpack5.i234 = extractelement <2 x double> %38, i32 1
  %39 = fadd double %.unpack.i231, %.unpack3.i233
  %40 = fadd double %.unpack2.i232, %.unpack5.i234
  %.not = icmp eq i32 %23, 31
  br i1 %.not, label %202, label %41

41:                                               ; preds = %11
  %42 = add nuw nsw i32 %27, 2
  tail call void @llvm.experimental.noalias.scope.decl(metadata !10)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !13)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !15)
  %43 = icmp samesign ult i32 %28, 62
  br i1 %43, label %44, label %81

44:                                               ; preds = %41
  %45 = zext nneg i32 %28 to i64
  %46 = getelementptr inbounds [16 x i8], ptr addrspace(1) %14, i64 %45
  %47 = load <2 x double>, ptr addrspace(1) %46, align 16, !invariant.load !9, !alias.scope !15, !noalias !17
  %.unpack.i19217 = extractelement <2 x double> %47, i32 0
  %.unpack2.i21218 = extractelement <2 x double> %47, i32 1
  tail call void @llvm.experimental.noalias.scope.decl(metadata !18)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !21)
  %48 = shl nuw nsw i32 %42, 6
  %49 = zext nneg i32 %48 to i64
  %50 = add i64 %49, %45
  %51 = getelementptr inbounds [16 x i8], ptr addrspace(1) %12, i64 %50
  %52 = getelementptr inbounds i8, ptr addrspace(1) %51, i64 32
  %53 = load <2 x double>, ptr addrspace(1) %52, align 16, !invariant.load !9, !alias.scope !23, !noalias !24
  %.unpack.i.i221 = extractelement <2 x double> %53, i32 0
  %.unpack2.i.i222 = extractelement <2 x double> %53, i32 1
  %54 = getelementptr inbounds i8, ptr addrspace(1) %37, i64 2048
  %55 = getelementptr inbounds i8, ptr addrspace(1) %54, i64 32
  %56 = load <2 x double>, ptr addrspace(1) %55, align 16, !invariant.load !9, !alias.scope !25, !noalias !26
  %.unpack3.i.i225 = extractelement <2 x double> %56, i32 0
  %.unpack5.i.i226 = extractelement <2 x double> %56, i32 1
  %57 = fadd double %.unpack.i.i221, %.unpack3.i.i225
  %58 = fadd double %.unpack2.i.i222, %.unpack5.i.i226
  %59 = fmul double %.unpack.i19217, %57
  %60 = fmul double %.unpack2.i21218, %58
  %61 = fmul double %.unpack2.i21218, %57
  %62 = fmul double %.unpack.i19217, %58
  %63 = fsub double %59, %60
  %64 = fadd double %61, %62
  tail call void @llvm.experimental.noalias.scope.decl(metadata !27)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !30)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !32)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !35)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !37)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !40)
  %65 = or disjoint i32 %48, %28
  %66 = zext nneg i32 %65 to i64
  %67 = getelementptr inbounds [16 x i8], ptr addrspace(1) %12, i64 %66
  %68 = load <2 x double>, ptr addrspace(1) %67, align 16, !invariant.load !9, !alias.scope !42, !noalias !43
  %.unpack.i.i.i.i219 = extractelement <2 x double> %68, i32 0
  %.unpack2.i.i.i.i220 = extractelement <2 x double> %68, i32 1
  %69 = getelementptr inbounds i8, ptr addrspace(1) %37, i64 32
  %70 = load <2 x double>, ptr addrspace(1) %69, align 16, !invariant.load !9, !alias.scope !44, !noalias !45
  %.unpack3.i.i.i.i223 = extractelement <2 x double> %70, i32 0
  %.unpack5.i.i.i.i224 = extractelement <2 x double> %70, i32 1
  %71 = fadd double %.unpack.i.i.i.i219, %.unpack3.i.i.i.i223
  %72 = fadd double %.unpack2.i.i.i.i220, %.unpack5.i.i.i.i224
  %73 = fmul double %71, -4.000000e+00
  %74 = fmul double %72, 0.000000e+00
  %75 = fsub double %73, %74
  %76 = fadd double %63, %75
  %77 = fmul double %71, 0.000000e+00
  %78 = fmul double %72, 4.000000e+00
  %79 = fsub double %77, %78
  %80 = fadd double %64, %79
  br label %fused_add_1_complex_29_20_clone_1.exit

81:                                               ; preds = %41
  tail call void @llvm.experimental.noalias.scope.decl(metadata !46)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !49)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !51)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !54)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !56)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !59)
  %82 = shl nuw nsw i32 %42, 6
  %83 = or disjoint i32 %82, %28
  %84 = zext nneg i32 %83 to i64
  %85 = getelementptr inbounds [16 x i8], ptr addrspace(1) %12, i64 %84
  %86 = load <2 x double>, ptr addrspace(1) %85, align 16, !invariant.load !9, !alias.scope !61, !noalias !62
  %.unpack.i.i.i9.i227 = extractelement <2 x double> %86, i32 0
  %.unpack2.i.i.i11.i228 = extractelement <2 x double> %86, i32 1
  %87 = getelementptr inbounds i8, ptr addrspace(1) %37, i64 32
  %88 = load <2 x double>, ptr addrspace(1) %87, align 16, !invariant.load !9, !alias.scope !63, !noalias !64
  %.unpack3.i.i.i12.i229 = extractelement <2 x double> %88, i32 0
  %.unpack5.i.i.i14.i230 = extractelement <2 x double> %88, i32 1
  %89 = fadd double %.unpack.i.i.i9.i227, %.unpack3.i.i.i12.i229
  %90 = fadd double %.unpack2.i.i.i11.i228, %.unpack5.i.i.i14.i230
  %91 = fmul double %89, -4.000000e+00
  %92 = fmul double %90, 0.000000e+00
  %93 = fsub double %91, %92
  %94 = fmul double %89, 0.000000e+00
  %95 = fmul double %90, 4.000000e+00
  %96 = fsub double %94, %95
  br label %fused_add_1_complex_29_20_clone_1.exit

fused_add_1_complex_29_20_clone_1.exit:           ; preds = %44, %81
  %97 = phi double [ %93, %81 ], [ %76, %44 ]
  %98 = phi double [ %96, %81 ], [ %80, %44 ]
  %99 = icmp ult i32 %28, 62
  %100 = zext nneg i32 %27 to i64
  %101 = getelementptr inbounds [16 x i8], ptr addrspace(1) %15, i64 %100
  %102 = load <2 x double>, ptr addrspace(1) %101, align 16, !invariant.load !9
  %.unpack215 = extractelement <2 x double> %102, i32 0
  %.unpack5216 = extractelement <2 x double> %102, i32 1
  %103 = fmul double %97, %.unpack215
  %104 = fmul double %98, %.unpack5216
  %105 = fmul double %98, %.unpack215
  %106 = fmul double %97, %.unpack5216
  %107 = fsub double %103, %104
  %108 = fadd double %105, %106
  %109 = load i8, ptr addrspace(1) %16, align 256, !invariant.load !9
  tail call void @llvm.experimental.noalias.scope.decl(metadata !65)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !68)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !70)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !72)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !75)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !77)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !79)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !82)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !84)
  br i1 %99, label %110, label %141

110:                                              ; preds = %fused_add_1_complex_29_20_clone_1.exit
  %111 = zext nneg i32 %28 to i64
  %112 = getelementptr inbounds [16 x i8], ptr addrspace(1) %14, i64 %111
  %113 = load <2 x double>, ptr addrspace(1) %112, align 16, !invariant.load !9, !alias.scope !86, !noalias !87
  %.unpack.i.i.i209 = extractelement <2 x double> %113, i32 0
  %.unpack2.i.i.i210 = extractelement <2 x double> %113, i32 1
  %114 = add nuw nsw i32 %28, 2
  tail call void @llvm.experimental.noalias.scope.decl(metadata !88)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !91)
  %115 = zext nneg i32 %29 to i64
  %116 = add i64 %115, %111
  %117 = getelementptr inbounds [16 x i8], ptr addrspace(1) %12, i64 %116
  %118 = getelementptr inbounds i8, ptr addrspace(1) %117, i64 32
  %119 = load <2 x double>, ptr addrspace(1) %118, align 16, !invariant.load !9, !alias.scope !93, !noalias !94
  %.unpack.i.i.i.i22211 = extractelement <2 x double> %119, i32 0
  %.unpack2.i.i.i.i24212 = extractelement <2 x double> %119, i32 1
  %120 = shl nuw nsw i32 %114, 6
  %121 = or disjoint i32 %120, %27
  %122 = zext nneg i32 %121 to i64
  %123 = getelementptr inbounds [16 x i8], ptr addrspace(1) %13, i64 %122
  %124 = load <2 x double>, ptr addrspace(1) %123, align 16, !invariant.load !9, !alias.scope !95, !noalias !96
  %.unpack3.i.i.i.i25213 = extractelement <2 x double> %124, i32 0
  %.unpack5.i.i.i.i27214 = extractelement <2 x double> %124, i32 1
  %125 = fadd double %.unpack.i.i.i.i22211, %.unpack3.i.i.i.i25213
  %126 = fadd double %.unpack2.i.i.i.i24212, %.unpack5.i.i.i.i27214
  %127 = fmul double %.unpack.i.i.i209, %125
  %128 = fmul double %.unpack2.i.i.i210, %126
  %129 = fmul double %.unpack2.i.i.i210, %125
  %130 = fmul double %.unpack.i.i.i209, %126
  %131 = fsub double %127, %128
  %132 = fadd double %129, %130
  %133 = fmul double %39, -4.000000e+00
  %134 = fmul double %40, 0.000000e+00
  %135 = fsub double %133, %134
  %136 = fadd double %135, %131
  %137 = fmul double %39, 0.000000e+00
  %138 = fmul double %40, 4.000000e+00
  %139 = fsub double %137, %138
  %140 = fadd double %139, %132
  br label %fused_add_1_real_46_5_clone_1.exit

141:                                              ; preds = %fused_add_1_complex_29_20_clone_1.exit
  %142 = fmul double %39, -4.000000e+00
  %143 = fmul double %40, 0.000000e+00
  %144 = fsub double %142, %143
  %145 = fmul double %39, 0.000000e+00
  %146 = fmul double %40, 4.000000e+00
  %147 = fsub double %145, %146
  br label %fused_add_1_real_46_5_clone_1.exit

fused_add_1_real_46_5_clone_1.exit:               ; preds = %110, %141
  %148 = phi double [ %144, %141 ], [ %136, %110 ]
  %149 = phi double [ %147, %141 ], [ %140, %110 ]
  %150 = icmp ult i32 %28, 62
  %151 = fmul double %148, 4.000000e+00
  %152 = fmul double %149, 0.000000e+00
  %153 = fsub double %152, %151
  %154 = fadd double %107, %153
  %155 = trunc i8 %109 to i1
  %156 = select i1 %155, double %154, double %153
  tail call void @llvm.experimental.noalias.scope.decl(metadata !97)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !100)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !102)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !104)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !107)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !109)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !111)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !114)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !116)
  br i1 %150, label %157, label %188

157:                                              ; preds = %fused_add_1_real_46_5_clone_1.exit
  %158 = zext nneg i32 %28 to i64
  %159 = getelementptr inbounds [16 x i8], ptr addrspace(1) %14, i64 %158
  %160 = load <2 x double>, ptr addrspace(1) %159, align 16, !invariant.load !9, !alias.scope !118, !noalias !119
  %.unpack.i.i.i34203 = extractelement <2 x double> %160, i32 0
  %.unpack2.i.i.i36204 = extractelement <2 x double> %160, i32 1
  %161 = add nuw nsw i32 %28, 2
  tail call void @llvm.experimental.noalias.scope.decl(metadata !120)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !123)
  %162 = zext nneg i32 %29 to i64
  %163 = add i64 %162, %158
  %164 = getelementptr inbounds [16 x i8], ptr addrspace(1) %12, i64 %163
  %165 = getelementptr inbounds i8, ptr addrspace(1) %164, i64 32
  %166 = load <2 x double>, ptr addrspace(1) %165, align 16, !invariant.load !9, !alias.scope !125, !noalias !126
  %.unpack.i.i.i.i37205 = extractelement <2 x double> %166, i32 0
  %.unpack2.i.i.i.i39206 = extractelement <2 x double> %166, i32 1
  %167 = shl nuw nsw i32 %161, 6
  %168 = or disjoint i32 %167, %27
  %169 = zext nneg i32 %168 to i64
  %170 = getelementptr inbounds [16 x i8], ptr addrspace(1) %13, i64 %169
  %171 = load <2 x double>, ptr addrspace(1) %170, align 16, !invariant.load !9, !alias.scope !127, !noalias !128
  %.unpack3.i.i.i.i40207 = extractelement <2 x double> %171, i32 0
  %.unpack5.i.i.i.i42208 = extractelement <2 x double> %171, i32 1
  %172 = fadd double %.unpack.i.i.i.i37205, %.unpack3.i.i.i.i40207
  %173 = fadd double %.unpack2.i.i.i.i39206, %.unpack5.i.i.i.i42208
  %174 = fmul double %.unpack.i.i.i34203, %172
  %175 = fmul double %.unpack2.i.i.i36204, %173
  %176 = fmul double %.unpack2.i.i.i36204, %172
  %177 = fmul double %.unpack.i.i.i34203, %173
  %178 = fsub double %174, %175
  %179 = fadd double %176, %177
  %180 = fmul double %39, -4.000000e+00
  %181 = fmul double %40, 0.000000e+00
  %182 = fsub double %180, %181
  %183 = fadd double %182, %178
  %184 = fmul double %39, 0.000000e+00
  %185 = fmul double %40, 4.000000e+00
  %186 = fsub double %184, %185
  %187 = fadd double %186, %179
  br label %fused_add_1_imag_37_5_clone_1.exit

188:                                              ; preds = %fused_add_1_real_46_5_clone_1.exit
  %189 = fmul double %39, -4.000000e+00
  %190 = fmul double %40, 0.000000e+00
  %191 = fsub double %189, %190
  %192 = fmul double %39, 0.000000e+00
  %193 = fmul double %40, 4.000000e+00
  %194 = fsub double %192, %193
  br label %fused_add_1_imag_37_5_clone_1.exit

fused_add_1_imag_37_5_clone_1.exit:               ; preds = %157, %188
  %195 = phi double [ %191, %188 ], [ %183, %157 ]
  %196 = phi double [ %194, %188 ], [ %187, %157 ]
  %197 = fmul double %196, 4.000000e+00
  %198 = fmul double %195, -0.000000e+00
  %199 = fsub double %198, %197
  %200 = fadd double %108, %199
  %201 = select i1 %155, double %200, double %199
  br label %291

202:                                              ; preds = %11
  tail call void @llvm.experimental.noalias.scope.decl(metadata !129)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !132)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !134)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !136)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !139)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !141)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !143)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !146)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !148)
  %203 = icmp samesign ult i32 %28, 62
  br i1 %203, label %204, label %235

204:                                              ; preds = %202
  %205 = zext nneg i32 %28 to i64
  %206 = getelementptr inbounds [16 x i8], ptr addrspace(1) %14, i64 %205
  %207 = load <2 x double>, ptr addrspace(1) %206, align 16, !invariant.load !9, !alias.scope !150, !noalias !151
  %.unpack.i.i.i55197 = extractelement <2 x double> %207, i32 0
  %.unpack2.i.i.i57198 = extractelement <2 x double> %207, i32 1
  %208 = add nuw nsw i32 %28, 2
  tail call void @llvm.experimental.noalias.scope.decl(metadata !152)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !155)
  %209 = zext nneg i32 %29 to i64
  %210 = add i64 %209, %205
  %211 = getelementptr inbounds [16 x i8], ptr addrspace(1) %12, i64 %210
  %212 = getelementptr inbounds i8, ptr addrspace(1) %211, i64 32
  %213 = load <2 x double>, ptr addrspace(1) %212, align 16, !invariant.load !9, !alias.scope !157, !noalias !158
  %.unpack.i.i.i.i58199 = extractelement <2 x double> %213, i32 0
  %.unpack2.i.i.i.i60200 = extractelement <2 x double> %213, i32 1
  %214 = shl nuw nsw i32 %208, 6
  %215 = or disjoint i32 %214, %27
  %216 = zext nneg i32 %215 to i64
  %217 = getelementptr inbounds [16 x i8], ptr addrspace(1) %13, i64 %216
  %218 = load <2 x double>, ptr addrspace(1) %217, align 16, !invariant.load !9, !alias.scope !159, !noalias !160
  %.unpack3.i.i.i.i61201 = extractelement <2 x double> %218, i32 0
  %.unpack5.i.i.i.i63202 = extractelement <2 x double> %218, i32 1
  %219 = fadd double %.unpack.i.i.i.i58199, %.unpack3.i.i.i.i61201
  %220 = fadd double %.unpack2.i.i.i.i60200, %.unpack5.i.i.i.i63202
  %221 = fmul double %.unpack.i.i.i55197, %219
  %222 = fmul double %.unpack2.i.i.i57198, %220
  %223 = fmul double %.unpack2.i.i.i57198, %219
  %224 = fmul double %.unpack.i.i.i55197, %220
  %225 = fsub double %221, %222
  %226 = fadd double %223, %224
  %227 = fmul double %39, -4.000000e+00
  %228 = fmul double %40, 0.000000e+00
  %229 = fsub double %227, %228
  %230 = fadd double %229, %225
  %231 = fmul double %39, 0.000000e+00
  %232 = fmul double %40, 4.000000e+00
  %233 = fsub double %231, %232
  %234 = fadd double %233, %226
  br label %fused_add_1_real_46_5_clone_1.exit70

235:                                              ; preds = %202
  %236 = fmul double %39, -4.000000e+00
  %237 = fmul double %40, 0.000000e+00
  %238 = fsub double %236, %237
  %239 = fmul double %39, 0.000000e+00
  %240 = fmul double %40, 4.000000e+00
  %241 = fsub double %239, %240
  br label %fused_add_1_real_46_5_clone_1.exit70

fused_add_1_real_46_5_clone_1.exit70:             ; preds = %204, %235
  %242 = phi double [ %238, %235 ], [ %230, %204 ]
  %243 = phi double [ %241, %235 ], [ %234, %204 ]
  %244 = icmp ult i32 %28, 62
  %245 = fmul double %242, 4.000000e+00
  %246 = fmul double %243, 0.000000e+00
  %247 = fsub double %246, %245
  tail call void @llvm.experimental.noalias.scope.decl(metadata !161)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !164)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !166)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !168)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !171)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !173)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !175)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !178)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !180)
  br i1 %244, label %248, label %279

248:                                              ; preds = %fused_add_1_real_46_5_clone_1.exit70
  %249 = zext nneg i32 %28 to i64
  %250 = getelementptr inbounds [16 x i8], ptr addrspace(1) %14, i64 %249
  %251 = load <2 x double>, ptr addrspace(1) %250, align 16, !invariant.load !9, !alias.scope !182, !noalias !183
  %.unpack.i.i.i77191 = extractelement <2 x double> %251, i32 0
  %.unpack2.i.i.i79192 = extractelement <2 x double> %251, i32 1
  %252 = add nuw nsw i32 %28, 2
  tail call void @llvm.experimental.noalias.scope.decl(metadata !184)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !187)
  %253 = zext nneg i32 %29 to i64
  %254 = add i64 %253, %249
  %255 = getelementptr inbounds [16 x i8], ptr addrspace(1) %12, i64 %254
  %256 = getelementptr inbounds i8, ptr addrspace(1) %255, i64 32
  %257 = load <2 x double>, ptr addrspace(1) %256, align 16, !invariant.load !9, !alias.scope !189, !noalias !190
  %.unpack.i.i.i.i80193 = extractelement <2 x double> %257, i32 0
  %.unpack2.i.i.i.i82194 = extractelement <2 x double> %257, i32 1
  %258 = shl nuw nsw i32 %252, 6
  %259 = or disjoint i32 %258, %27
  %260 = zext nneg i32 %259 to i64
  %261 = getelementptr inbounds [16 x i8], ptr addrspace(1) %13, i64 %260
  %262 = load <2 x double>, ptr addrspace(1) %261, align 16, !invariant.load !9, !alias.scope !191, !noalias !192
  %.unpack3.i.i.i.i83195 = extractelement <2 x double> %262, i32 0
  %.unpack5.i.i.i.i85196 = extractelement <2 x double> %262, i32 1
  %263 = fadd double %.unpack.i.i.i.i80193, %.unpack3.i.i.i.i83195
  %264 = fadd double %.unpack2.i.i.i.i82194, %.unpack5.i.i.i.i85196
  %265 = fmul double %.unpack.i.i.i77191, %263
  %266 = fmul double %.unpack2.i.i.i79192, %264
  %267 = fmul double %.unpack2.i.i.i79192, %263
  %268 = fmul double %.unpack.i.i.i77191, %264
  %269 = fsub double %265, %266
  %270 = fadd double %267, %268
  %271 = fmul double %39, -4.000000e+00
  %272 = fmul double %40, 0.000000e+00
  %273 = fsub double %271, %272
  %274 = fadd double %273, %269
  %275 = fmul double %39, 0.000000e+00
  %276 = fmul double %40, 4.000000e+00
  %277 = fsub double %275, %276
  %278 = fadd double %277, %270
  br label %fused_add_1_imag_37_5_clone_1.exit92

279:                                              ; preds = %fused_add_1_real_46_5_clone_1.exit70
  %280 = fmul double %39, -4.000000e+00
  %281 = fmul double %40, 0.000000e+00
  %282 = fsub double %280, %281
  %283 = fmul double %39, 0.000000e+00
  %284 = fmul double %40, 4.000000e+00
  %285 = fsub double %283, %284
  br label %fused_add_1_imag_37_5_clone_1.exit92

fused_add_1_imag_37_5_clone_1.exit92:             ; preds = %248, %279
  %286 = phi double [ %282, %279 ], [ %274, %248 ]
  %287 = phi double [ %285, %279 ], [ %278, %248 ]
  %288 = fmul double %287, 4.000000e+00
  %289 = fmul double %286, -0.000000e+00
  %290 = fsub double %289, %288
  br label %291

291:                                              ; preds = %fused_add_1_imag_37_5_clone_1.exit92, %fused_add_1_imag_37_5_clone_1.exit
  %292 = phi double [ %247, %fused_add_1_imag_37_5_clone_1.exit92 ], [ %156, %fused_add_1_imag_37_5_clone_1.exit ]
  %293 = phi double [ %290, %fused_add_1_imag_37_5_clone_1.exit92 ], [ %201, %fused_add_1_imag_37_5_clone_1.exit ]
  %.not6 = icmp eq i32 %27, 63
  br i1 %.not6, label %329, label %294

294:                                              ; preds = %291
  %.not7 = icmp eq i32 %28, 63
  br i1 %.not7, label %317, label %295

295:                                              ; preds = %294
  %296 = add nuw nsw i32 %27, 1
  %297 = zext nneg i32 %28 to i64
  %298 = getelementptr inbounds [16 x i8], ptr addrspace(1) %17, i64 %297
  %299 = load <2 x double>, ptr addrspace(1) %298, align 16, !invariant.load !9
  %.unpack8185 = extractelement <2 x double> %299, i32 0
  %.unpack10186 = extractelement <2 x double> %299, i32 1
  tail call void @llvm.experimental.noalias.scope.decl(metadata !193)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !196)
  %300 = shl nuw nsw i32 %296, 6
  %301 = zext nneg i32 %300 to i64
  %302 = add i64 %301, %297
  %303 = getelementptr inbounds [16 x i8], ptr addrspace(1) %12, i64 %302
  %304 = getelementptr inbounds i8, ptr addrspace(1) %303, i64 16
  %305 = load <2 x double>, ptr addrspace(1) %304, align 16, !invariant.load !9, !alias.scope !193, !noalias !196
  %.unpack.i93187 = extractelement <2 x double> %305, i32 0
  %.unpack2.i95188 = extractelement <2 x double> %305, i32 1
  %306 = getelementptr inbounds i8, ptr addrspace(1) %37, i64 1024
  %307 = getelementptr inbounds i8, ptr addrspace(1) %306, i64 16
  %308 = load <2 x double>, ptr addrspace(1) %307, align 16, !invariant.load !9, !alias.scope !196, !noalias !193
  %.unpack3.i96189 = extractelement <2 x double> %308, i32 0
  %.unpack5.i98190 = extractelement <2 x double> %308, i32 1
  %309 = fadd double %.unpack.i93187, %.unpack3.i96189
  %310 = fadd double %.unpack2.i95188, %.unpack5.i98190
  %311 = fmul double %.unpack8185, %309
  %312 = fmul double %.unpack10186, %310
  %313 = fmul double %.unpack10186, %309
  %314 = fmul double %.unpack8185, %310
  %315 = fsub double %311, %312
  %316 = fadd double %313, %314
  br label %317

317:                                              ; preds = %295, %294
  %318 = phi double [ %315, %295 ], [ 0.000000e+00, %294 ]
  %319 = phi double [ %316, %295 ], [ 0.000000e+00, %294 ]
  %320 = zext nneg i32 %27 to i64
  %321 = getelementptr inbounds [16 x i8], ptr addrspace(1) %18, i64 %320
  %322 = load <2 x double>, ptr addrspace(1) %321, align 16, !invariant.load !9
  %.unpack11183 = extractelement <2 x double> %322, i32 0
  %.unpack13184 = extractelement <2 x double> %322, i32 1
  %323 = fmul double %318, %.unpack11183
  %324 = fmul double %319, %.unpack13184
  %325 = fmul double %319, %.unpack11183
  %326 = fmul double %318, %.unpack13184
  %327 = fsub double %323, %324
  %328 = fadd double %325, %326
  br label %329

329:                                              ; preds = %317, %291
  %330 = phi double [ %327, %317 ], [ 0.000000e+00, %291 ]
  %331 = phi double [ %328, %317 ], [ 0.000000e+00, %291 ]
  %332 = icmp samesign ult i32 %28, 62
  br i1 %332, label %333, label %fused_add_1_add_334_3_clone_1.exit138

333:                                              ; preds = %329
  %334 = add nuw nsw i32 %28, 2
  %335 = zext nneg i32 %28 to i64
  %336 = getelementptr inbounds [8 x i8], ptr addrspace(1) %19, i64 %335
  %337 = getelementptr inbounds i8, ptr addrspace(1) %336, i64 1040
  %338 = load double, ptr addrspace(1) %337, align 8, !invariant.load !9, !alias.scope !198, !noalias !205
  %339 = fmul double %338, 5.000000e-01
  %340 = fsub double 0.000000e+00, %339
  %341 = fmul double %338, 0.000000e+00
  %342 = fadd double %341, 0.000000e+00
  tail call void @llvm.experimental.noalias.scope.decl(metadata !207)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !210)
  %343 = zext nneg i32 %29 to i64
  %344 = add i64 %343, %335
  %345 = getelementptr inbounds [16 x i8], ptr addrspace(1) %12, i64 %344
  %346 = getelementptr inbounds i8, ptr addrspace(1) %345, i64 32
  %347 = load <2 x double>, ptr addrspace(1) %346, align 16, !invariant.load !9, !alias.scope !207, !noalias !210
  %.unpack.i99175 = extractelement <2 x double> %347, i32 0
  %.unpack2.i101176 = extractelement <2 x double> %347, i32 1
  %348 = shl nuw nsw i32 %334, 6
  %349 = or disjoint i32 %348, %27
  %350 = zext nneg i32 %349 to i64
  %351 = getelementptr inbounds [16 x i8], ptr addrspace(1) %13, i64 %350
  %352 = load <2 x double>, ptr addrspace(1) %351, align 16, !invariant.load !9, !alias.scope !210, !noalias !207
  %.unpack3.i102177 = extractelement <2 x double> %352, i32 0
  %.unpack5.i104178 = extractelement <2 x double> %352, i32 1
  %353 = fadd double %.unpack.i99175, %.unpack3.i102177
  %354 = fadd double %.unpack2.i101176, %.unpack5.i104178
  %355 = fmul double %340, %353
  %356 = fmul double %342, %354
  %357 = fmul double %342, %353
  %358 = fmul double %340, %354
  %359 = fsub double %355, %356
  %360 = fadd double %357, %358
  tail call void @llvm.experimental.noalias.scope.decl(metadata !212)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !215)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !217)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !219)
  %361 = icmp samesign ugt i32 %28, 1
  br i1 %361, label %362, label %fused_add_1_add_334_3_clone_1.exit

362:                                              ; preds = %333
  %363 = add nsw i32 %28, -2
  tail call void @llvm.experimental.noalias.scope.decl(metadata !221)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !224)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !226)
  %364 = zext nneg i32 %363 to i64
  %365 = getelementptr inbounds [8 x i8], ptr addrspace(1) %19, i64 %364
  %366 = load double, ptr addrspace(1) %365, align 8, !invariant.load !9, !alias.scope !228, !noalias !239
  %367 = fmul double %366, -5.000000e-01
  %368 = fmul double %366, 0.000000e+00
  tail call void @llvm.experimental.noalias.scope.decl(metadata !241)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !244)
  %369 = add nuw nsw i32 %29, %363
  %370 = zext nneg i32 %369 to i64
  %371 = getelementptr inbounds [16 x i8], ptr addrspace(1) %12, i64 %370
  %372 = load <2 x double>, ptr addrspace(1) %371, align 16, !invariant.load !9, !alias.scope !246, !noalias !247
  %.unpack.i.i.i105171 = extractelement <2 x double> %372, i32 0
  %.unpack2.i.i.i107172 = extractelement <2 x double> %372, i32 1
  %373 = shl nuw nsw i32 %363, 6
  %374 = or disjoint i32 %373, %27
  %375 = zext nneg i32 %374 to i64
  %376 = getelementptr inbounds [16 x i8], ptr addrspace(1) %13, i64 %375
  %377 = load <2 x double>, ptr addrspace(1) %376, align 16, !invariant.load !9, !alias.scope !248, !noalias !249
  %.unpack3.i.i.i173 = extractelement <2 x double> %377, i32 0
  %.unpack5.i.i.i174 = extractelement <2 x double> %377, i32 1
  %378 = fadd double %.unpack.i.i.i105171, %.unpack3.i.i.i173
  %379 = fadd double %.unpack2.i.i.i107172, %.unpack5.i.i.i174
  %380 = fmul double %367, %378
  %381 = fmul double %368, %379
  %382 = fsub double %380, %381
  br label %fused_add_1_add_334_3_clone_1.exit

fused_add_1_add_334_3_clone_1.exit:               ; preds = %333, %362
  %383 = phi double [ %382, %362 ], [ 0.000000e+00, %333 ]
  %384 = icmp ugt i32 %28, 1
  tail call void @llvm.experimental.noalias.scope.decl(metadata !250)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !253)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !255)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !258)
  %385 = getelementptr inbounds i8, ptr addrspace(1) %336, i64 512
  %386 = load double, ptr addrspace(1) %385, align 8, !invariant.load !9, !alias.scope !260, !noalias !267
  %387 = fmul double %386, 5.000000e-01
  %388 = fsub double 0.000000e+00, %387
  %389 = getelementptr inbounds [8 x i8], ptr addrspace(1) %20, i64 %335
  %390 = load double, ptr addrspace(1) %389, align 8, !invariant.load !9, !alias.scope !270, !noalias !275
  %391 = fmul double %386, 0.000000e+00
  %392 = fadd double %390, %391
  %393 = fmul double %388, %39
  %394 = fmul double %392, %40
  %395 = fsub double %393, %394
  %396 = fadd double %383, %395
  %397 = fadd double %359, %396
  tail call void @llvm.experimental.noalias.scope.decl(metadata !278)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !281)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !283)
  br i1 %384, label %398, label %fused_add_1_add_336_3_clone_1.exit

398:                                              ; preds = %fused_add_1_add_334_3_clone_1.exit
  %399 = add nsw i32 %28, -2
  tail call void @llvm.experimental.noalias.scope.decl(metadata !285)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !288)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !290)
  %400 = zext nneg i32 %399 to i64
  %401 = getelementptr inbounds [8 x i8], ptr addrspace(1) %19, i64 %400
  %402 = load double, ptr addrspace(1) %401, align 8, !invariant.load !9, !alias.scope !292, !noalias !303
  %403 = fmul double %402, 5.000000e-01
  %404 = fmul double %402, 0.000000e+00
  tail call void @llvm.experimental.noalias.scope.decl(metadata !306)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !309)
  %405 = add nuw nsw i32 %29, %399
  %406 = zext nneg i32 %405 to i64
  %407 = getelementptr inbounds [16 x i8], ptr addrspace(1) %12, i64 %406
  %408 = load <2 x double>, ptr addrspace(1) %407, align 16, !invariant.load !9, !alias.scope !311, !noalias !312
  %.unpack.i.i.i117167 = extractelement <2 x double> %408, i32 0
  %.unpack2.i.i.i119168 = extractelement <2 x double> %408, i32 1
  %409 = shl nuw nsw i32 %399, 6
  %410 = or disjoint i32 %409, %27
  %411 = zext nneg i32 %410 to i64
  %412 = getelementptr inbounds [16 x i8], ptr addrspace(1) %13, i64 %411
  %413 = load <2 x double>, ptr addrspace(1) %412, align 16, !invariant.load !9, !alias.scope !313, !noalias !314
  %.unpack3.i.i.i120169 = extractelement <2 x double> %413, i32 0
  %.unpack5.i.i.i122170 = extractelement <2 x double> %413, i32 1
  %414 = fadd double %.unpack.i.i.i117167, %.unpack3.i.i.i120169
  %415 = fadd double %.unpack2.i.i.i119168, %.unpack5.i.i.i122170
  %416 = fmul double %404, %414
  %417 = fmul double %403, %415
  %418 = fsub double %416, %417
  br label %fused_add_1_add_336_3_clone_1.exit

fused_add_1_add_336_3_clone_1.exit:               ; preds = %fused_add_1_add_334_3_clone_1.exit, %398
  %419 = phi double [ %418, %398 ], [ 0.000000e+00, %fused_add_1_add_334_3_clone_1.exit ]
  %420 = fmul double %392, %39
  %421 = fmul double %388, %40
  %422 = fadd double %420, %421
  %423 = fadd double %419, %422
  %424 = fadd double %360, %423
  br label %467

fused_add_1_add_334_3_clone_1.exit138:            ; preds = %329
  tail call void @llvm.experimental.noalias.scope.decl(metadata !315)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !318)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !320)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !322)
  %425 = add nsw i32 %28, -2
  tail call void @llvm.experimental.noalias.scope.decl(metadata !324)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !327)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !329)
  %426 = zext nneg i32 %425 to i64
  %427 = getelementptr inbounds [8 x i8], ptr addrspace(1) %19, i64 %426
  %428 = load double, ptr addrspace(1) %427, align 8, !invariant.load !9, !alias.scope !331, !noalias !342
  %429 = fmul double %428, -5.000000e-01
  %430 = fmul double %428, 0.000000e+00
  tail call void @llvm.experimental.noalias.scope.decl(metadata !344)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !347)
  %431 = add nuw nsw i32 %29, %425
  %432 = zext nneg i32 %431 to i64
  %433 = getelementptr inbounds [16 x i8], ptr addrspace(1) %12, i64 %432
  %434 = load <2 x double>, ptr addrspace(1) %433, align 16, !invariant.load !9, !alias.scope !349, !noalias !350
  %.unpack.i.i.i132179 = extractelement <2 x double> %434, i32 0
  %.unpack2.i.i.i134180 = extractelement <2 x double> %434, i32 1
  %435 = shl nuw nsw i32 %425, 6
  %436 = or disjoint i32 %435, %27
  %437 = zext nneg i32 %436 to i64
  %438 = getelementptr inbounds [16 x i8], ptr addrspace(1) %13, i64 %437
  %439 = load <2 x double>, ptr addrspace(1) %438, align 16, !invariant.load !9, !alias.scope !351, !noalias !352
  %.unpack3.i.i.i135181 = extractelement <2 x double> %439, i32 0
  %.unpack5.i.i.i137182 = extractelement <2 x double> %439, i32 1
  %440 = fadd double %.unpack.i.i.i132179, %.unpack3.i.i.i135181
  %441 = fadd double %.unpack2.i.i.i134180, %.unpack5.i.i.i137182
  %442 = fmul double %429, %440
  %443 = fmul double %430, %441
  %444 = fsub double %442, %443
  tail call void @llvm.experimental.noalias.scope.decl(metadata !353)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !356)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !358)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !361)
  %445 = zext nneg i32 %28 to i64
  %446 = getelementptr inbounds [8 x i8], ptr addrspace(1) %19, i64 %445
  %447 = getelementptr inbounds i8, ptr addrspace(1) %446, i64 512
  %448 = load double, ptr addrspace(1) %447, align 8, !invariant.load !9, !alias.scope !363, !noalias !370
  %449 = fmul double %448, 5.000000e-01
  %450 = fsub double 0.000000e+00, %449
  %451 = getelementptr inbounds [8 x i8], ptr addrspace(1) %20, i64 %445
  %452 = load double, ptr addrspace(1) %451, align 8, !invariant.load !9, !alias.scope !373, !noalias !378
  %453 = fmul double %448, 0.000000e+00
  %454 = fadd double %452, %453
  %455 = fmul double %450, %39
  %456 = fmul double %454, %40
  %457 = fsub double %455, %456
  %458 = fadd double %444, %457
  %459 = fmul double %428, 5.000000e-01
  %460 = fmul double %430, %440
  %461 = fmul double %459, %441
  %462 = fsub double %460, %461
  %463 = fmul double %454, %39
  %464 = fmul double %450, %40
  %465 = fadd double %463, %464
  %466 = fadd double %462, %465
  br label %467

467:                                              ; preds = %fused_add_1_add_334_3_clone_1.exit138, %fused_add_1_add_336_3_clone_1.exit
  %468 = phi double [ %458, %fused_add_1_add_334_3_clone_1.exit138 ], [ %397, %fused_add_1_add_336_3_clone_1.exit ]
  %469 = phi double [ %466, %fused_add_1_add_334_3_clone_1.exit138 ], [ %424, %fused_add_1_add_336_3_clone_1.exit ]
  %470 = fadd double %292, %330
  %471 = fadd double %293, %331
  %472 = fmul double %470, 5.000000e-01
  %473 = fmul double %471, 0.000000e+00
  %474 = fmul double %471, 5.000000e-01
  %475 = fmul double %470, 0.000000e+00
  %476 = fsub double %472, %473
  %477 = fadd double %475, %474
  %478 = fadd double %476, %468
  %479 = fadd double %477, %469
  %480 = shl nuw nsw i32 %23, 7
  %481 = or disjoint i32 %480, %24
  %482 = zext nneg i32 %481 to i64
  %483 = getelementptr inbounds [16 x i8], ptr addrspace(1) %21, i64 %482
  %484 = insertelement <2 x double> poison, double %39, i32 0
  %485 = insertelement <2 x double> %484, double %40, i32 1
  store <2 x double> %485, ptr addrspace(1) %483, align 16
  %486 = getelementptr inbounds [16 x i8], ptr addrspace(1) %22, i64 %482
  %487 = insertelement <2 x double> poison, double %478, i32 0
  %488 = insertelement <2 x double> %487, double %479, i32 1
  store <2 x double> %488, ptr addrspace(1) %486, align 16
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.experimental.noalias.scope.decl(metadata) #2

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite) "nvvm.reqntid"="128,1,1" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{i32 0, i32 32}
!3 = !{i32 0, i32 128}
!4 = !{!5}
!5 = distinct !{!5, !6, !"fused_add_1_add_333_1: argument 0"}
!6 = distinct !{!6, !"fused_add_1_add_333_1"}
!7 = !{!8}
!8 = distinct !{!8, !6, !"fused_add_1_add_333_1: argument 1"}
!9 = !{}
!10 = !{!11}
!11 = distinct !{!11, !12, !"fused_add_1_complex_29_20_clone_1: argument 0"}
!12 = distinct !{!12, !"fused_add_1_complex_29_20_clone_1"}
!13 = !{!14}
!14 = distinct !{!14, !12, !"fused_add_1_complex_29_20_clone_1: argument 1"}
!15 = !{!16}
!16 = distinct !{!16, !12, !"fused_add_1_complex_29_20_clone_1: argument 2"}
!17 = !{!11, !14}
!18 = !{!19}
!19 = distinct !{!19, !20, !"fused_add_1_add_333_1: argument 0"}
!20 = distinct !{!20, !"fused_add_1_add_333_1"}
!21 = !{!22}
!22 = distinct !{!22, !20, !"fused_add_1_add_333_1: argument 1"}
!23 = !{!19, !11}
!24 = !{!22, !14, !16}
!25 = !{!22, !14}
!26 = !{!19, !11, !16}
!27 = !{!28}
!28 = distinct !{!28, !29, !"fused_add_1_real_44_5_clone_1: argument 0"}
!29 = distinct !{!29, !"fused_add_1_real_44_5_clone_1"}
!30 = !{!31}
!31 = distinct !{!31, !29, !"fused_add_1_real_44_5_clone_1: argument 1"}
!32 = !{!33}
!33 = distinct !{!33, !34, !"fused_add_1_multiply_70_9_clone_1: argument 0"}
!34 = distinct !{!34, !"fused_add_1_multiply_70_9_clone_1"}
!35 = !{!36}
!36 = distinct !{!36, !34, !"fused_add_1_multiply_70_9_clone_1: argument 1"}
!37 = !{!38}
!38 = distinct !{!38, !39, !"fused_add_1_add_333_1: argument 0"}
!39 = distinct !{!39, !"fused_add_1_add_333_1"}
!40 = !{!41}
!41 = distinct !{!41, !39, !"fused_add_1_add_333_1: argument 1"}
!42 = !{!38, !33, !28, !11}
!43 = !{!41, !36, !31, !14, !16}
!44 = !{!41, !36, !31, !14}
!45 = !{!38, !33, !28, !11, !16}
!46 = !{!47}
!47 = distinct !{!47, !48, !"fused_add_1_real_44_5_clone_1: argument 0"}
!48 = distinct !{!48, !"fused_add_1_real_44_5_clone_1"}
!49 = !{!50}
!50 = distinct !{!50, !48, !"fused_add_1_real_44_5_clone_1: argument 1"}
!51 = !{!52}
!52 = distinct !{!52, !53, !"fused_add_1_multiply_70_9_clone_1: argument 0"}
!53 = distinct !{!53, !"fused_add_1_multiply_70_9_clone_1"}
!54 = !{!55}
!55 = distinct !{!55, !53, !"fused_add_1_multiply_70_9_clone_1: argument 1"}
!56 = !{!57}
!57 = distinct !{!57, !58, !"fused_add_1_add_333_1: argument 0"}
!58 = distinct !{!58, !"fused_add_1_add_333_1"}
!59 = !{!60}
!60 = distinct !{!60, !58, !"fused_add_1_add_333_1: argument 1"}
!61 = !{!57, !52, !47, !11}
!62 = !{!60, !55, !50, !14, !16}
!63 = !{!60, !55, !50, !14}
!64 = !{!57, !52, !47, !11, !16}
!65 = !{!66}
!66 = distinct !{!66, !67, !"fused_add_1_real_46_5_clone_1: argument 0"}
!67 = distinct !{!67, !"fused_add_1_real_46_5_clone_1"}
!68 = !{!69}
!69 = distinct !{!69, !67, !"fused_add_1_real_46_5_clone_1: argument 1"}
!70 = !{!71}
!71 = distinct !{!71, !67, !"fused_add_1_real_46_5_clone_1: argument 2"}
!72 = !{!73}
!73 = distinct !{!73, !74, !"fused_add_1_multiply_72_13_clone_1: argument 0"}
!74 = distinct !{!74, !"fused_add_1_multiply_72_13_clone_1"}
!75 = !{!76}
!76 = distinct !{!76, !74, !"fused_add_1_multiply_72_13_clone_1: argument 1"}
!77 = !{!78}
!78 = distinct !{!78, !74, !"fused_add_1_multiply_72_13_clone_1: argument 2"}
!79 = !{!80}
!80 = distinct !{!80, !81, !"fused_add_1_complex_29_20_clone_1: argument 0"}
!81 = distinct !{!81, !"fused_add_1_complex_29_20_clone_1"}
!82 = !{!83}
!83 = distinct !{!83, !81, !"fused_add_1_complex_29_20_clone_1: argument 1"}
!84 = !{!85}
!85 = distinct !{!85, !81, !"fused_add_1_complex_29_20_clone_1: argument 2"}
!86 = !{!85, !78, !71}
!87 = !{!80, !83, !73, !76, !66, !69}
!88 = !{!89}
!89 = distinct !{!89, !90, !"fused_add_1_add_333_1: argument 0"}
!90 = distinct !{!90, !"fused_add_1_add_333_1"}
!91 = !{!92}
!92 = distinct !{!92, !90, !"fused_add_1_add_333_1: argument 1"}
!93 = !{!89, !80, !73, !66}
!94 = !{!92, !83, !85, !76, !78, !69, !71}
!95 = !{!92, !83, !76, !69}
!96 = !{!89, !80, !85, !73, !78, !66, !71}
!97 = !{!98}
!98 = distinct !{!98, !99, !"fused_add_1_imag_37_5_clone_1: argument 0"}
!99 = distinct !{!99, !"fused_add_1_imag_37_5_clone_1"}
!100 = !{!101}
!101 = distinct !{!101, !99, !"fused_add_1_imag_37_5_clone_1: argument 1"}
!102 = !{!103}
!103 = distinct !{!103, !99, !"fused_add_1_imag_37_5_clone_1: argument 2"}
!104 = !{!105}
!105 = distinct !{!105, !106, !"fused_add_1_multiply_72_13_clone_1: argument 0"}
!106 = distinct !{!106, !"fused_add_1_multiply_72_13_clone_1"}
!107 = !{!108}
!108 = distinct !{!108, !106, !"fused_add_1_multiply_72_13_clone_1: argument 1"}
!109 = !{!110}
!110 = distinct !{!110, !106, !"fused_add_1_multiply_72_13_clone_1: argument 2"}
!111 = !{!112}
!112 = distinct !{!112, !113, !"fused_add_1_complex_29_20_clone_1: argument 0"}
!113 = distinct !{!113, !"fused_add_1_complex_29_20_clone_1"}
!114 = !{!115}
!115 = distinct !{!115, !113, !"fused_add_1_complex_29_20_clone_1: argument 1"}
!116 = !{!117}
!117 = distinct !{!117, !113, !"fused_add_1_complex_29_20_clone_1: argument 2"}
!118 = !{!117, !110, !103}
!119 = !{!112, !115, !105, !108, !98, !101}
!120 = !{!121}
!121 = distinct !{!121, !122, !"fused_add_1_add_333_1: argument 0"}
!122 = distinct !{!122, !"fused_add_1_add_333_1"}
!123 = !{!124}
!124 = distinct !{!124, !122, !"fused_add_1_add_333_1: argument 1"}
!125 = !{!121, !112, !105, !98}
!126 = !{!124, !115, !117, !108, !110, !101, !103}
!127 = !{!124, !115, !108, !101}
!128 = !{!121, !112, !117, !105, !110, !98, !103}
!129 = !{!130}
!130 = distinct !{!130, !131, !"fused_add_1_real_46_5_clone_1: argument 0"}
!131 = distinct !{!131, !"fused_add_1_real_46_5_clone_1"}
!132 = !{!133}
!133 = distinct !{!133, !131, !"fused_add_1_real_46_5_clone_1: argument 1"}
!134 = !{!135}
!135 = distinct !{!135, !131, !"fused_add_1_real_46_5_clone_1: argument 2"}
!136 = !{!137}
!137 = distinct !{!137, !138, !"fused_add_1_multiply_72_13_clone_1: argument 0"}
!138 = distinct !{!138, !"fused_add_1_multiply_72_13_clone_1"}
!139 = !{!140}
!140 = distinct !{!140, !138, !"fused_add_1_multiply_72_13_clone_1: argument 1"}
!141 = !{!142}
!142 = distinct !{!142, !138, !"fused_add_1_multiply_72_13_clone_1: argument 2"}
!143 = !{!144}
!144 = distinct !{!144, !145, !"fused_add_1_complex_29_20_clone_1: argument 0"}
!145 = distinct !{!145, !"fused_add_1_complex_29_20_clone_1"}
!146 = !{!147}
!147 = distinct !{!147, !145, !"fused_add_1_complex_29_20_clone_1: argument 1"}
!148 = !{!149}
!149 = distinct !{!149, !145, !"fused_add_1_complex_29_20_clone_1: argument 2"}
!150 = !{!149, !142, !135}
!151 = !{!144, !147, !137, !140, !130, !133}
!152 = !{!153}
!153 = distinct !{!153, !154, !"fused_add_1_add_333_1: argument 0"}
!154 = distinct !{!154, !"fused_add_1_add_333_1"}
!155 = !{!156}
!156 = distinct !{!156, !154, !"fused_add_1_add_333_1: argument 1"}
!157 = !{!153, !144, !137, !130}
!158 = !{!156, !147, !149, !140, !142, !133, !135}
!159 = !{!156, !147, !140, !133}
!160 = !{!153, !144, !149, !137, !142, !130, !135}
!161 = !{!162}
!162 = distinct !{!162, !163, !"fused_add_1_imag_37_5_clone_1: argument 0"}
!163 = distinct !{!163, !"fused_add_1_imag_37_5_clone_1"}
!164 = !{!165}
!165 = distinct !{!165, !163, !"fused_add_1_imag_37_5_clone_1: argument 1"}
!166 = !{!167}
!167 = distinct !{!167, !163, !"fused_add_1_imag_37_5_clone_1: argument 2"}
!168 = !{!169}
!169 = distinct !{!169, !170, !"fused_add_1_multiply_72_13_clone_1: argument 0"}
!170 = distinct !{!170, !"fused_add_1_multiply_72_13_clone_1"}
!171 = !{!172}
!172 = distinct !{!172, !170, !"fused_add_1_multiply_72_13_clone_1: argument 1"}
!173 = !{!174}
!174 = distinct !{!174, !170, !"fused_add_1_multiply_72_13_clone_1: argument 2"}
!175 = !{!176}
!176 = distinct !{!176, !177, !"fused_add_1_complex_29_20_clone_1: argument 0"}
!177 = distinct !{!177, !"fused_add_1_complex_29_20_clone_1"}
!178 = !{!179}
!179 = distinct !{!179, !177, !"fused_add_1_complex_29_20_clone_1: argument 1"}
!180 = !{!181}
!181 = distinct !{!181, !177, !"fused_add_1_complex_29_20_clone_1: argument 2"}
!182 = !{!181, !174, !167}
!183 = !{!176, !179, !169, !172, !162, !165}
!184 = !{!185}
!185 = distinct !{!185, !186, !"fused_add_1_add_333_1: argument 0"}
!186 = distinct !{!186, !"fused_add_1_add_333_1"}
!187 = !{!188}
!188 = distinct !{!188, !186, !"fused_add_1_add_333_1: argument 1"}
!189 = !{!185, !176, !169, !162}
!190 = !{!188, !179, !181, !172, !174, !165, !167}
!191 = !{!188, !179, !172, !165}
!192 = !{!185, !176, !181, !169, !174, !162, !167}
!193 = !{!194}
!194 = distinct !{!194, !195, !"fused_add_1_add_333_1: argument 0"}
!195 = distinct !{!195, !"fused_add_1_add_333_1"}
!196 = !{!197}
!197 = distinct !{!197, !195, !"fused_add_1_add_333_1: argument 1"}
!198 = !{!199, !201, !203}
!199 = distinct !{!199, !200, !"fused_add_1_mul_333_3_clone_1: argument 0"}
!200 = distinct !{!200, !"fused_add_1_mul_333_3_clone_1"}
!201 = distinct !{!201, !202, !"fused_add_1_scatter_add_663_3_clone_1: argument 0"}
!202 = distinct !{!202, !"fused_add_1_scatter_add_663_3_clone_1"}
!203 = distinct !{!203, !204, !"fused_add_1_scatter_add_627_7_clone_1: argument 1"}
!204 = distinct !{!204, !"fused_add_1_scatter_add_627_7_clone_1"}
!205 = !{!206}
!206 = distinct !{!206, !204, !"fused_add_1_scatter_add_627_7_clone_1: argument 0"}
!207 = !{!208}
!208 = distinct !{!208, !209, !"fused_add_1_add_333_1: argument 0"}
!209 = distinct !{!209, !"fused_add_1_add_333_1"}
!210 = !{!211}
!211 = distinct !{!211, !209, !"fused_add_1_add_333_1: argument 1"}
!212 = !{!213}
!213 = distinct !{!213, !214, !"fused_add_1_add_334_3_clone_1: argument 0"}
!214 = distinct !{!214, !"fused_add_1_add_334_3_clone_1"}
!215 = !{!216}
!216 = distinct !{!216, !214, !"fused_add_1_add_334_3_clone_1: argument 1"}
!217 = !{!218}
!218 = distinct !{!218, !214, !"fused_add_1_add_334_3_clone_1: argument 2"}
!219 = !{!220}
!220 = distinct !{!220, !214, !"fused_add_1_add_334_3_clone_1: argument 3"}
!221 = !{!222}
!222 = distinct !{!222, !223, !"fused_add_1_multiply_67_13_clone_1: argument 0"}
!223 = distinct !{!223, !"fused_add_1_multiply_67_13_clone_1"}
!224 = !{!225}
!225 = distinct !{!225, !223, !"fused_add_1_multiply_67_13_clone_1: argument 1"}
!226 = !{!227}
!227 = distinct !{!227, !223, !"fused_add_1_multiply_67_13_clone_1: argument 2"}
!228 = !{!229, !231, !233, !235, !237, !227, !220}
!229 = distinct !{!229, !230, !"fused_add_1_mul_333_3_clone_1: argument 0"}
!230 = distinct !{!230, !"fused_add_1_mul_333_3_clone_1"}
!231 = distinct !{!231, !232, !"fused_add_1_scatter_add_663_3_clone_1: argument 0"}
!232 = distinct !{!232, !"fused_add_1_scatter_add_663_3_clone_1"}
!233 = distinct !{!233, !234, !"fused_add_1_dynamic_update_slice_197_9_clone_1: argument 0"}
!234 = distinct !{!234, !"fused_add_1_dynamic_update_slice_197_9_clone_1"}
!235 = distinct !{!235, !236, !"fused_add_1_dynamic_update_slice_198_8_clone_1: argument 0"}
!236 = distinct !{!236, !"fused_add_1_dynamic_update_slice_198_8_clone_1"}
!237 = distinct !{!237, !238, !"fused_add_1_scatter_add_627_7_clone_1: argument 1"}
!238 = distinct !{!238, !"fused_add_1_scatter_add_627_7_clone_1"}
!239 = !{!240, !222, !225, !213, !216, !218}
!240 = distinct !{!240, !238, !"fused_add_1_scatter_add_627_7_clone_1: argument 0"}
!241 = !{!242}
!242 = distinct !{!242, !243, !"fused_add_1_add_333_1: argument 0"}
!243 = distinct !{!243, !"fused_add_1_add_333_1"}
!244 = !{!245}
!245 = distinct !{!245, !243, !"fused_add_1_add_333_1: argument 1"}
!246 = !{!242, !222, !213}
!247 = !{!245, !225, !227, !216, !218, !220}
!248 = !{!245, !225, !216}
!249 = !{!242, !222, !227, !213, !218, !220}
!250 = !{!251}
!251 = distinct !{!251, !252, !"fused_add_1_multiply_68_13_clone_1: argument 2"}
!252 = distinct !{!252, !"fused_add_1_multiply_68_13_clone_1"}
!253 = !{!254}
!254 = distinct !{!254, !252, !"fused_add_1_multiply_68_13_clone_1: argument 3"}
!255 = !{!256}
!256 = distinct !{!256, !257, !"fused_add_1_scatter_add_627_7_clone_1: argument 0"}
!257 = distinct !{!257, !"fused_add_1_scatter_add_627_7_clone_1"}
!258 = !{!259}
!259 = distinct !{!259, !257, !"fused_add_1_scatter_add_627_7_clone_1: argument 1"}
!260 = !{!261, !263, !265, !259, !254, !220}
!261 = distinct !{!261, !262, !"fused_add_1_mul_333_3_clone_1: argument 0"}
!262 = distinct !{!262, !"fused_add_1_mul_333_3_clone_1"}
!263 = distinct !{!263, !264, !"fused_add_1_scatter_add_663_3_clone_1: argument 0"}
!264 = distinct !{!264, !"fused_add_1_scatter_add_663_3_clone_1"}
!265 = distinct !{!265, !266, !"fused_add_1_dynamic_update_slice_198_8_clone_1: argument 0"}
!266 = distinct !{!266, !"fused_add_1_dynamic_update_slice_198_8_clone_1"}
!267 = !{!256, !268, !269, !251, !213, !216, !218}
!268 = distinct !{!268, !252, !"fused_add_1_multiply_68_13_clone_1: argument 0"}
!269 = distinct !{!269, !252, !"fused_add_1_multiply_68_13_clone_1: argument 1"}
!270 = !{!271, !273, !256, !251, !218}
!271 = distinct !{!271, !272, !"fused_add_1_dynamic_update_slice_200_9_clone_1: argument 0"}
!272 = distinct !{!272, !"fused_add_1_dynamic_update_slice_200_9_clone_1"}
!273 = distinct !{!273, !274, !"fused_add_1_dynamic_update_slice_201_8_clone_1: argument 0"}
!274 = distinct !{!274, !"fused_add_1_dynamic_update_slice_201_8_clone_1"}
!275 = !{!276, !277, !259, !268, !269, !254, !213, !216, !220}
!276 = distinct !{!276, !272, !"fused_add_1_dynamic_update_slice_200_9_clone_1: argument 1"}
!277 = distinct !{!277, !274, !"fused_add_1_dynamic_update_slice_201_8_clone_1: argument 1"}
!278 = !{!279}
!279 = distinct !{!279, !280, !"fused_add_1_add_336_3_clone_1: argument 0"}
!280 = distinct !{!280, !"fused_add_1_add_336_3_clone_1"}
!281 = !{!282}
!282 = distinct !{!282, !280, !"fused_add_1_add_336_3_clone_1: argument 1"}
!283 = !{!284}
!284 = distinct !{!284, !280, !"fused_add_1_add_336_3_clone_1: argument 3"}
!285 = !{!286}
!286 = distinct !{!286, !287, !"fused_add_1_multiply_67_13_clone_1: argument 0"}
!287 = distinct !{!287, !"fused_add_1_multiply_67_13_clone_1"}
!288 = !{!289}
!289 = distinct !{!289, !287, !"fused_add_1_multiply_67_13_clone_1: argument 1"}
!290 = !{!291}
!291 = distinct !{!291, !287, !"fused_add_1_multiply_67_13_clone_1: argument 2"}
!292 = !{!293, !295, !297, !299, !301, !291, !284}
!293 = distinct !{!293, !294, !"fused_add_1_mul_333_3_clone_1: argument 0"}
!294 = distinct !{!294, !"fused_add_1_mul_333_3_clone_1"}
!295 = distinct !{!295, !296, !"fused_add_1_scatter_add_663_3_clone_1: argument 0"}
!296 = distinct !{!296, !"fused_add_1_scatter_add_663_3_clone_1"}
!297 = distinct !{!297, !298, !"fused_add_1_dynamic_update_slice_197_9_clone_1: argument 0"}
!298 = distinct !{!298, !"fused_add_1_dynamic_update_slice_197_9_clone_1"}
!299 = distinct !{!299, !300, !"fused_add_1_dynamic_update_slice_198_8_clone_1: argument 0"}
!300 = distinct !{!300, !"fused_add_1_dynamic_update_slice_198_8_clone_1"}
!301 = distinct !{!301, !302, !"fused_add_1_scatter_add_627_7_clone_1: argument 1"}
!302 = distinct !{!302, !"fused_add_1_scatter_add_627_7_clone_1"}
!303 = !{!304, !286, !289, !279, !282, !305}
!304 = distinct !{!304, !302, !"fused_add_1_scatter_add_627_7_clone_1: argument 0"}
!305 = distinct !{!305, !280, !"fused_add_1_add_336_3_clone_1: argument 2"}
!306 = !{!307}
!307 = distinct !{!307, !308, !"fused_add_1_add_333_1: argument 0"}
!308 = distinct !{!308, !"fused_add_1_add_333_1"}
!309 = !{!310}
!310 = distinct !{!310, !308, !"fused_add_1_add_333_1: argument 1"}
!311 = !{!307, !286, !279}
!312 = !{!310, !289, !291, !282, !305, !284}
!313 = !{!310, !289, !282}
!314 = !{!307, !286, !291, !279, !305, !284}
!315 = !{!316}
!316 = distinct !{!316, !317, !"fused_add_1_add_334_3_clone_1: argument 0"}
!317 = distinct !{!317, !"fused_add_1_add_334_3_clone_1"}
!318 = !{!319}
!319 = distinct !{!319, !317, !"fused_add_1_add_334_3_clone_1: argument 1"}
!320 = !{!321}
!321 = distinct !{!321, !317, !"fused_add_1_add_334_3_clone_1: argument 2"}
!322 = !{!323}
!323 = distinct !{!323, !317, !"fused_add_1_add_334_3_clone_1: argument 3"}
!324 = !{!325}
!325 = distinct !{!325, !326, !"fused_add_1_multiply_67_13_clone_1: argument 0"}
!326 = distinct !{!326, !"fused_add_1_multiply_67_13_clone_1"}
!327 = !{!328}
!328 = distinct !{!328, !326, !"fused_add_1_multiply_67_13_clone_1: argument 1"}
!329 = !{!330}
!330 = distinct !{!330, !326, !"fused_add_1_multiply_67_13_clone_1: argument 2"}
!331 = !{!332, !334, !336, !338, !340, !330, !323}
!332 = distinct !{!332, !333, !"fused_add_1_mul_333_3_clone_1: argument 0"}
!333 = distinct !{!333, !"fused_add_1_mul_333_3_clone_1"}
!334 = distinct !{!334, !335, !"fused_add_1_scatter_add_663_3_clone_1: argument 0"}
!335 = distinct !{!335, !"fused_add_1_scatter_add_663_3_clone_1"}
!336 = distinct !{!336, !337, !"fused_add_1_dynamic_update_slice_197_9_clone_1: argument 0"}
!337 = distinct !{!337, !"fused_add_1_dynamic_update_slice_197_9_clone_1"}
!338 = distinct !{!338, !339, !"fused_add_1_dynamic_update_slice_198_8_clone_1: argument 0"}
!339 = distinct !{!339, !"fused_add_1_dynamic_update_slice_198_8_clone_1"}
!340 = distinct !{!340, !341, !"fused_add_1_scatter_add_627_7_clone_1: argument 1"}
!341 = distinct !{!341, !"fused_add_1_scatter_add_627_7_clone_1"}
!342 = !{!343, !325, !328, !316, !319, !321}
!343 = distinct !{!343, !341, !"fused_add_1_scatter_add_627_7_clone_1: argument 0"}
!344 = !{!345}
!345 = distinct !{!345, !346, !"fused_add_1_add_333_1: argument 0"}
!346 = distinct !{!346, !"fused_add_1_add_333_1"}
!347 = !{!348}
!348 = distinct !{!348, !346, !"fused_add_1_add_333_1: argument 1"}
!349 = !{!345, !325, !316}
!350 = !{!348, !328, !330, !319, !321, !323}
!351 = !{!348, !328, !319}
!352 = !{!345, !325, !330, !316, !321, !323}
!353 = !{!354}
!354 = distinct !{!354, !355, !"fused_add_1_multiply_68_13_clone_1: argument 2"}
!355 = distinct !{!355, !"fused_add_1_multiply_68_13_clone_1"}
!356 = !{!357}
!357 = distinct !{!357, !355, !"fused_add_1_multiply_68_13_clone_1: argument 3"}
!358 = !{!359}
!359 = distinct !{!359, !360, !"fused_add_1_scatter_add_627_7_clone_1: argument 0"}
!360 = distinct !{!360, !"fused_add_1_scatter_add_627_7_clone_1"}
!361 = !{!362}
!362 = distinct !{!362, !360, !"fused_add_1_scatter_add_627_7_clone_1: argument 1"}
!363 = !{!364, !366, !368, !362, !357, !323}
!364 = distinct !{!364, !365, !"fused_add_1_mul_333_3_clone_1: argument 0"}
!365 = distinct !{!365, !"fused_add_1_mul_333_3_clone_1"}
!366 = distinct !{!366, !367, !"fused_add_1_scatter_add_663_3_clone_1: argument 0"}
!367 = distinct !{!367, !"fused_add_1_scatter_add_663_3_clone_1"}
!368 = distinct !{!368, !369, !"fused_add_1_dynamic_update_slice_198_8_clone_1: argument 0"}
!369 = distinct !{!369, !"fused_add_1_dynamic_update_slice_198_8_clone_1"}
!370 = !{!359, !371, !372, !354, !316, !319, !321}
!371 = distinct !{!371, !355, !"fused_add_1_multiply_68_13_clone_1: argument 0"}
!372 = distinct !{!372, !355, !"fused_add_1_multiply_68_13_clone_1: argument 1"}
!373 = !{!374, !376, !359, !354, !321}
!374 = distinct !{!374, !375, !"fused_add_1_dynamic_update_slice_200_9_clone_1: argument 0"}
!375 = distinct !{!375, !"fused_add_1_dynamic_update_slice_200_9_clone_1"}
!376 = distinct !{!376, !377, !"fused_add_1_dynamic_update_slice_201_8_clone_1: argument 0"}
!377 = distinct !{!377, !"fused_add_1_dynamic_update_slice_201_8_clone_1"}
!378 = !{!379, !380, !362, !371, !372, !357, !316, !319, !323}
!379 = distinct !{!379, !375, !"fused_add_1_dynamic_update_slice_200_9_clone_1: argument 1"}
!380 = distinct !{!380, !377, !"fused_add_1_dynamic_update_slice_201_8_clone_1: argument 1"}
