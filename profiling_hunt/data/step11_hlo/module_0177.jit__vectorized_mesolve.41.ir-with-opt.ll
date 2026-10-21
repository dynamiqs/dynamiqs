; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define ptx_kernel void @input_add_fusion(ptr noalias nofree readonly align 256 captures(none) dereferenceable(65536) %0, ptr noalias nofree readonly align 256 captures(none) dereferenceable(65536) %1, ptr noalias nofree readonly align 16 captures(none) dereferenceable(1024) %2, ptr noalias nofree readonly align 256 captures(none) dereferenceable(1) %3, ptr noalias nofree readonly align 16 captures(none) dereferenceable(2048) %4, ptr noalias nofree readonly align 256 captures(none) dereferenceable(3072) %5, ptr noalias nofree readonly align 16 captures(none) dereferenceable(1024) %6, ptr noalias nofree writeonly align 256 captures(none) dereferenceable(65536) %7, ptr noalias nofree writeonly align 256 captures(none) dereferenceable(65536) %8) local_unnamed_addr #0 {
  %10 = addrspacecast ptr %0 to ptr addrspace(1)
  %11 = addrspacecast ptr %1 to ptr addrspace(1)
  %12 = addrspacecast ptr %4 to ptr addrspace(1)
  %13 = addrspacecast ptr %3 to ptr addrspace(1)
  %14 = addrspacecast ptr %2 to ptr addrspace(1)
  %15 = addrspacecast ptr %5 to ptr addrspace(1)
  %16 = addrspacecast ptr %6 to ptr addrspace(1)
  %17 = addrspacecast ptr %7 to ptr addrspace(1)
  %18 = addrspacecast ptr %8 to ptr addrspace(1)
  %19 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !2
  %20 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !3
  %21 = shl nuw nsw i32 %19, 1
  %22 = lshr i32 %20, 6
  %23 = or disjoint i32 %21, %22
  %24 = and i32 %20, 63
  tail call void @llvm.experimental.noalias.scope.decl(metadata !4)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !7)
  %25 = shl nuw nsw i32 %23, 6
  %26 = or disjoint i32 %25, %24
  %27 = zext nneg i32 %26 to i64
  %28 = getelementptr inbounds [16 x i8], ptr addrspace(1) %10, i64 %27
  %29 = load <2 x double>, ptr addrspace(1) %28, align 16, !invariant.load !9, !alias.scope !4, !noalias !7
  %.unpack.i331 = extractelement <2 x double> %29, i32 0
  %.unpack2.i332 = extractelement <2 x double> %29, i32 1
  %30 = shl nuw nsw i32 %24, 6
  %31 = or disjoint i32 %30, %23
  %32 = zext nneg i32 %31 to i64
  %33 = getelementptr inbounds [16 x i8], ptr addrspace(1) %11, i64 %32
  %34 = load <2 x double>, ptr addrspace(1) %33, align 16, !invariant.load !9, !alias.scope !7, !noalias !4
  %.unpack3.i333 = extractelement <2 x double> %34, i32 0
  %.unpack5.i334 = extractelement <2 x double> %34, i32 1
  %35 = fadd double %.unpack.i331, %.unpack3.i333
  %36 = fadd double %.unpack2.i332, %.unpack5.i334
  %.not = icmp eq i32 %19, 31
  br i1 %.not, label %207, label %37

37:                                               ; preds = %9
  %38 = add nuw nsw i32 %23, 2
  tail call void @llvm.experimental.noalias.scope.decl(metadata !10)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !13)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !15)
  %39 = icmp samesign ult i32 %24, 62
  br i1 %39, label %40, label %79

40:                                               ; preds = %37
  %41 = zext nneg i32 %24 to i64
  %42 = getelementptr inbounds [16 x i8], ptr addrspace(1) %12, i64 %41
  %43 = getelementptr inbounds i8, ptr addrspace(1) %42, i64 1056
  %44 = load <2 x double>, ptr addrspace(1) %43, align 16, !invariant.load !9, !alias.scope !15, !noalias !17
  %.unpack.i16315 = extractelement <2 x double> %44, i32 0
  %.unpack2.i18316 = extractelement <2 x double> %44, i32 1
  tail call void @llvm.experimental.noalias.scope.decl(metadata !18)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !21)
  %45 = shl nuw nsw i32 %38, 6
  %46 = zext nneg i32 %45 to i64
  %47 = add i64 %46, %41
  %48 = getelementptr inbounds [16 x i8], ptr addrspace(1) %10, i64 %47
  %49 = getelementptr inbounds i8, ptr addrspace(1) %48, i64 32
  %50 = load <2 x double>, ptr addrspace(1) %49, align 16, !invariant.load !9, !alias.scope !23, !noalias !24
  %.unpack.i.i319 = extractelement <2 x double> %50, i32 0
  %.unpack2.i.i320 = extractelement <2 x double> %50, i32 1
  %51 = getelementptr inbounds i8, ptr addrspace(1) %33, i64 2048
  %52 = getelementptr inbounds i8, ptr addrspace(1) %51, i64 32
  %53 = load <2 x double>, ptr addrspace(1) %52, align 16, !invariant.load !9, !alias.scope !25, !noalias !26
  %.unpack3.i.i323 = extractelement <2 x double> %53, i32 0
  %.unpack5.i.i324 = extractelement <2 x double> %53, i32 1
  %54 = fadd double %.unpack.i.i319, %.unpack3.i.i323
  %55 = fadd double %.unpack2.i.i320, %.unpack5.i.i324
  %56 = fmul double %.unpack.i16315, %54
  %57 = fmul double %.unpack2.i18316, %55
  %58 = fmul double %.unpack2.i18316, %54
  %59 = fmul double %.unpack.i16315, %55
  %60 = fsub double %56, %57
  %61 = fadd double %58, %59
  tail call void @llvm.experimental.noalias.scope.decl(metadata !27)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !30)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !32)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !34)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !37)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !39)
  %62 = load <2 x double>, ptr addrspace(1) %42, align 16, !invariant.load !9, !alias.scope !41, !noalias !42
  %.unpack.i.i.i313 = extractelement <2 x double> %62, i32 0
  %.unpack2.i.i.i314 = extractelement <2 x double> %62, i32 1
  tail call void @llvm.experimental.noalias.scope.decl(metadata !43)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !46)
  %63 = or disjoint i32 %45, %24
  %64 = zext nneg i32 %63 to i64
  %65 = getelementptr inbounds [16 x i8], ptr addrspace(1) %10, i64 %64
  %66 = load <2 x double>, ptr addrspace(1) %65, align 16, !invariant.load !9, !alias.scope !48, !noalias !49
  %.unpack.i.i.i.i317 = extractelement <2 x double> %66, i32 0
  %.unpack2.i.i.i.i318 = extractelement <2 x double> %66, i32 1
  %67 = getelementptr inbounds i8, ptr addrspace(1) %33, i64 32
  %68 = load <2 x double>, ptr addrspace(1) %67, align 16, !invariant.load !9, !alias.scope !50, !noalias !51
  %.unpack3.i.i.i.i321 = extractelement <2 x double> %68, i32 0
  %.unpack5.i.i.i.i322 = extractelement <2 x double> %68, i32 1
  %69 = fadd double %.unpack.i.i.i.i317, %.unpack3.i.i.i.i321
  %70 = fadd double %.unpack2.i.i.i.i318, %.unpack5.i.i.i.i322
  %71 = fmul double %.unpack.i.i.i313, %69
  %72 = fmul double %.unpack2.i.i.i314, %70
  %73 = fsub double %71, %72
  %74 = fadd double %60, %73
  %75 = fmul double %.unpack2.i.i.i314, %69
  %76 = fmul double %.unpack.i.i.i313, %70
  %77 = fadd double %75, %76
  %78 = fadd double %61, %77
  br label %fused_add_complex_36_16_clone_1.exit

79:                                               ; preds = %37
  tail call void @llvm.experimental.noalias.scope.decl(metadata !52)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !55)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !57)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !59)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !62)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !64)
  %80 = zext nneg i32 %24 to i64
  %81 = getelementptr inbounds [16 x i8], ptr addrspace(1) %12, i64 %80
  %82 = load <2 x double>, ptr addrspace(1) %81, align 16, !invariant.load !9, !alias.scope !66, !noalias !67
  %.unpack.i.i12.i325 = extractelement <2 x double> %82, i32 0
  %.unpack2.i.i14.i326 = extractelement <2 x double> %82, i32 1
  tail call void @llvm.experimental.noalias.scope.decl(metadata !68)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !71)
  %83 = shl nuw nsw i32 %38, 6
  %84 = or disjoint i32 %83, %24
  %85 = zext nneg i32 %84 to i64
  %86 = getelementptr inbounds [16 x i8], ptr addrspace(1) %10, i64 %85
  %87 = load <2 x double>, ptr addrspace(1) %86, align 16, !invariant.load !9, !alias.scope !73, !noalias !74
  %.unpack.i.i.i15.i327 = extractelement <2 x double> %87, i32 0
  %.unpack2.i.i.i17.i328 = extractelement <2 x double> %87, i32 1
  %88 = getelementptr inbounds i8, ptr addrspace(1) %33, i64 32
  %89 = load <2 x double>, ptr addrspace(1) %88, align 16, !invariant.load !9, !alias.scope !75, !noalias !76
  %.unpack3.i.i.i18.i329 = extractelement <2 x double> %89, i32 0
  %.unpack5.i.i.i20.i330 = extractelement <2 x double> %89, i32 1
  %90 = fadd double %.unpack.i.i.i15.i327, %.unpack3.i.i.i18.i329
  %91 = fadd double %.unpack2.i.i.i17.i328, %.unpack5.i.i.i20.i330
  %92 = fmul double %.unpack.i.i12.i325, %90
  %93 = fmul double %.unpack2.i.i14.i326, %91
  %94 = fsub double %92, %93
  %95 = fmul double %.unpack2.i.i14.i326, %90
  %96 = fmul double %.unpack.i.i12.i325, %91
  %97 = fadd double %95, %96
  br label %fused_add_complex_36_16_clone_1.exit

fused_add_complex_36_16_clone_1.exit:             ; preds = %79, %40
  %.unpack2.i.i.i.i.i51 = phi double [ %.unpack2.i.i14.i326, %79 ], [ %.unpack2.i.i.i314, %40 ]
  %.unpack.i.i.i.i.i49 = phi double [ %.unpack.i.i12.i325, %79 ], [ %.unpack.i.i.i313, %40 ]
  %98 = phi double [ %94, %79 ], [ %74, %40 ]
  %99 = phi double [ %97, %79 ], [ %78, %40 ]
  %100 = icmp ult i32 %24, 62
  %101 = zext nneg i32 %23 to i64
  %102 = getelementptr [16 x i8], ptr addrspace(1) %12, i64 %101
  %103 = getelementptr i8, ptr addrspace(1) %102, i64 1056
  %104 = load <2 x double>, ptr addrspace(1) %103, align 16, !invariant.load !9, !alias.scope !77
  %.unpack4.i311 = extractelement <2 x double> %104, i32 0
  %.unpack6.i312 = extractelement <2 x double> %104, i32 1
  %105 = fmul double %98, %.unpack4.i311
  %106 = fmul double %99, %.unpack4.i311
  %107 = fmul double %99, %.unpack6.i312
  %108 = fadd double %105, %107
  %109 = fmul double %98, %.unpack6.i312
  %110 = fsub double %106, %109
  %111 = load i8, ptr addrspace(1) %13, align 256, !invariant.load !9
  tail call void @llvm.experimental.noalias.scope.decl(metadata !80)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !83)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !85)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !87)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !90)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !92)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !94)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !97)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !99)
  br i1 %100, label %112, label %144

112:                                              ; preds = %fused_add_complex_36_16_clone_1.exit
  %113 = add nuw nsw i32 %24, 2
  %114 = zext nneg i32 %24 to i64
  %115 = getelementptr inbounds [16 x i8], ptr addrspace(1) %12, i64 %114
  %116 = getelementptr inbounds i8, ptr addrspace(1) %115, i64 1056
  %117 = load <2 x double>, ptr addrspace(1) %116, align 16, !invariant.load !9, !alias.scope !101, !noalias !102
  %.unpack.i1.i.i305 = extractelement <2 x double> %117, i32 0
  %.unpack2.i.i.i21306 = extractelement <2 x double> %117, i32 1
  tail call void @llvm.experimental.noalias.scope.decl(metadata !103)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !106)
  %118 = zext nneg i32 %25 to i64
  %119 = add i64 %118, %114
  %120 = getelementptr inbounds [16 x i8], ptr addrspace(1) %10, i64 %119
  %121 = getelementptr inbounds i8, ptr addrspace(1) %120, i64 32
  %122 = load <2 x double>, ptr addrspace(1) %121, align 16, !invariant.load !9, !alias.scope !108, !noalias !109
  %.unpack.i.i.i.i22307 = extractelement <2 x double> %122, i32 0
  %.unpack2.i.i.i.i24308 = extractelement <2 x double> %122, i32 1
  %123 = shl nuw nsw i32 %113, 6
  %124 = or disjoint i32 %123, %23
  %125 = zext nneg i32 %124 to i64
  %126 = getelementptr inbounds [16 x i8], ptr addrspace(1) %11, i64 %125
  %127 = load <2 x double>, ptr addrspace(1) %126, align 16, !invariant.load !9, !alias.scope !110, !noalias !111
  %.unpack3.i.i.i.i25309 = extractelement <2 x double> %127, i32 0
  %.unpack5.i.i.i.i27310 = extractelement <2 x double> %127, i32 1
  %128 = fadd double %.unpack.i.i.i.i22307, %.unpack3.i.i.i.i25309
  %129 = fadd double %.unpack2.i.i.i.i24308, %.unpack5.i.i.i.i27310
  %130 = fmul double %.unpack.i1.i.i305, %128
  %131 = fmul double %.unpack2.i.i.i21306, %129
  %132 = fmul double %.unpack2.i.i.i21306, %128
  %133 = fmul double %.unpack.i1.i.i305, %129
  %134 = fsub double %130, %131
  %135 = fadd double %132, %133
  %136 = fmul double %35, %.unpack.i.i.i.i.i49
  %137 = fmul double %36, %.unpack2.i.i.i.i.i51
  %138 = fsub double %136, %137
  %139 = fadd double %134, %138
  %140 = fmul double %35, %.unpack2.i.i.i.i.i51
  %141 = fmul double %36, %.unpack.i.i.i.i.i49
  %142 = fadd double %141, %140
  %143 = fadd double %135, %142
  br label %fused_add_real_53_5_clone_1.exit

144:                                              ; preds = %fused_add_complex_36_16_clone_1.exit
  %145 = fmul double %35, %.unpack.i.i.i.i.i49
  %146 = fmul double %36, %.unpack2.i.i.i.i.i51
  %147 = fsub double %145, %146
  %148 = fmul double %35, %.unpack2.i.i.i.i.i51
  %149 = fmul double %36, %.unpack.i.i.i.i.i49
  %150 = fadd double %149, %148
  br label %fused_add_real_53_5_clone_1.exit

fused_add_real_53_5_clone_1.exit:                 ; preds = %112, %144
  %151 = phi double [ %147, %144 ], [ %139, %112 ]
  %152 = phi double [ %150, %144 ], [ %143, %112 ]
  %153 = icmp ult i32 %24, 62
  %154 = load <2 x double>, ptr addrspace(1) %102, align 16, !invariant.load !9, !alias.scope !112, !noalias !115
  %.unpack.i.i.i19303 = extractelement <2 x double> %154, i32 0
  %.unpack3.i.i.i304 = extractelement <2 x double> %154, i32 1
  %155 = fmul double %151, %.unpack.i.i.i19303
  %156 = fmul double %152, %.unpack3.i.i.i304
  %157 = fadd double %156, %155
  %158 = fadd double %108, %157
  %159 = trunc i8 %111 to i1
  %160 = select i1 %159, double %158, double %157
  tail call void @llvm.experimental.noalias.scope.decl(metadata !116)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !119)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !121)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !123)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !126)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !128)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !130)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !133)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !135)
  br i1 %153, label %161, label %193

161:                                              ; preds = %fused_add_real_53_5_clone_1.exit
  %162 = add nuw nsw i32 %24, 2
  %163 = zext nneg i32 %24 to i64
  %164 = getelementptr inbounds [16 x i8], ptr addrspace(1) %12, i64 %163
  %165 = getelementptr inbounds i8, ptr addrspace(1) %164, i64 1056
  %166 = load <2 x double>, ptr addrspace(1) %165, align 16, !invariant.load !9, !alias.scope !137, !noalias !138
  %.unpack.i1.i.i40297 = extractelement <2 x double> %166, i32 0
  %.unpack2.i.i.i42298 = extractelement <2 x double> %166, i32 1
  tail call void @llvm.experimental.noalias.scope.decl(metadata !139)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !142)
  %167 = zext nneg i32 %25 to i64
  %168 = add i64 %167, %163
  %169 = getelementptr inbounds [16 x i8], ptr addrspace(1) %10, i64 %168
  %170 = getelementptr inbounds i8, ptr addrspace(1) %169, i64 32
  %171 = load <2 x double>, ptr addrspace(1) %170, align 16, !invariant.load !9, !alias.scope !144, !noalias !145
  %.unpack.i.i.i.i43299 = extractelement <2 x double> %171, i32 0
  %.unpack2.i.i.i.i45300 = extractelement <2 x double> %171, i32 1
  %172 = shl nuw nsw i32 %162, 6
  %173 = or disjoint i32 %172, %23
  %174 = zext nneg i32 %173 to i64
  %175 = getelementptr inbounds [16 x i8], ptr addrspace(1) %11, i64 %174
  %176 = load <2 x double>, ptr addrspace(1) %175, align 16, !invariant.load !9, !alias.scope !146, !noalias !147
  %.unpack3.i.i.i.i46301 = extractelement <2 x double> %176, i32 0
  %.unpack5.i.i.i.i48302 = extractelement <2 x double> %176, i32 1
  %177 = fadd double %.unpack.i.i.i.i43299, %.unpack3.i.i.i.i46301
  %178 = fadd double %.unpack2.i.i.i.i45300, %.unpack5.i.i.i.i48302
  %179 = fmul double %.unpack.i1.i.i40297, %177
  %180 = fmul double %.unpack2.i.i.i42298, %178
  %181 = fmul double %.unpack2.i.i.i42298, %177
  %182 = fmul double %.unpack.i1.i.i40297, %178
  %183 = fsub double %179, %180
  %184 = fadd double %181, %182
  %185 = fmul double %35, %.unpack.i.i.i.i.i49
  %186 = fmul double %36, %.unpack2.i.i.i.i.i51
  %187 = fsub double %185, %186
  %188 = fadd double %183, %187
  %189 = fmul double %35, %.unpack2.i.i.i.i.i51
  %190 = fmul double %36, %.unpack.i.i.i.i.i49
  %191 = fadd double %190, %189
  %192 = fadd double %184, %191
  br label %fused_add_imag_43_5_clone_1.exit

193:                                              ; preds = %fused_add_real_53_5_clone_1.exit
  %194 = fmul double %35, %.unpack.i.i.i.i.i49
  %195 = fmul double %36, %.unpack2.i.i.i.i.i51
  %196 = fsub double %194, %195
  %197 = fmul double %35, %.unpack2.i.i.i.i.i51
  %198 = fmul double %36, %.unpack.i.i.i.i.i49
  %199 = fadd double %198, %197
  br label %fused_add_imag_43_5_clone_1.exit

fused_add_imag_43_5_clone_1.exit:                 ; preds = %161, %193
  %200 = phi double [ %196, %193 ], [ %188, %161 ]
  %201 = phi double [ %199, %193 ], [ %192, %161 ]
  %202 = fmul double %.unpack.i.i.i19303, %201
  %203 = fmul double %.unpack3.i.i.i304, %200
  %204 = fsub double %202, %203
  %205 = fadd double %110, %204
  %206 = select i1 %159, double %205, double %204
  br label %305

207:                                              ; preds = %9
  tail call void @llvm.experimental.noalias.scope.decl(metadata !148)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !151)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !153)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !155)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !158)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !160)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !162)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !165)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !167)
  %208 = icmp samesign ult i32 %24, 62
  br i1 %208, label %209, label %242

209:                                              ; preds = %207
  %210 = add nuw nsw i32 %24, 2
  %211 = zext nneg i32 %24 to i64
  %212 = getelementptr inbounds [16 x i8], ptr addrspace(1) %12, i64 %211
  %213 = getelementptr inbounds i8, ptr addrspace(1) %212, i64 1056
  %214 = load <2 x double>, ptr addrspace(1) %213, align 16, !invariant.load !9, !alias.scope !169, !noalias !170
  %.unpack.i1.i.i70289 = extractelement <2 x double> %214, i32 0
  %.unpack2.i.i.i72290 = extractelement <2 x double> %214, i32 1
  tail call void @llvm.experimental.noalias.scope.decl(metadata !171)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !174)
  %215 = zext nneg i32 %25 to i64
  %216 = add i64 %215, %211
  %217 = getelementptr inbounds [16 x i8], ptr addrspace(1) %10, i64 %216
  %218 = getelementptr inbounds i8, ptr addrspace(1) %217, i64 32
  %219 = load <2 x double>, ptr addrspace(1) %218, align 16, !invariant.load !9, !alias.scope !176, !noalias !177
  %.unpack.i.i.i.i73291 = extractelement <2 x double> %219, i32 0
  %.unpack2.i.i.i.i75292 = extractelement <2 x double> %219, i32 1
  %220 = shl nuw nsw i32 %210, 6
  %221 = or disjoint i32 %220, %23
  %222 = zext nneg i32 %221 to i64
  %223 = getelementptr inbounds [16 x i8], ptr addrspace(1) %11, i64 %222
  %224 = load <2 x double>, ptr addrspace(1) %223, align 16, !invariant.load !9, !alias.scope !178, !noalias !179
  %.unpack3.i.i.i.i76293 = extractelement <2 x double> %224, i32 0
  %.unpack5.i.i.i.i78294 = extractelement <2 x double> %224, i32 1
  %225 = fadd double %.unpack.i.i.i.i73291, %.unpack3.i.i.i.i76293
  %226 = fadd double %.unpack2.i.i.i.i75292, %.unpack5.i.i.i.i78294
  %227 = fmul double %.unpack.i1.i.i70289, %225
  %228 = fmul double %.unpack2.i.i.i72290, %226
  %229 = fmul double %.unpack2.i.i.i72290, %225
  %230 = fmul double %.unpack.i1.i.i70289, %226
  %231 = fsub double %227, %228
  %232 = fadd double %229, %230
  %233 = load <2 x double>, ptr addrspace(1) %212, align 16, !invariant.load !9, !alias.scope !180, !noalias !185
  %.unpack.i.i.i.i.i79287 = extractelement <2 x double> %233, i32 0
  %.unpack2.i.i.i.i.i81288 = extractelement <2 x double> %233, i32 1
  %234 = fmul double %35, %.unpack.i.i.i.i.i79287
  %235 = fmul double %36, %.unpack2.i.i.i.i.i81288
  %236 = fsub double %234, %235
  %237 = fadd double %231, %236
  %238 = fmul double %35, %.unpack2.i.i.i.i.i81288
  %239 = fmul double %36, %.unpack.i.i.i.i.i79287
  %240 = fadd double %239, %238
  %241 = fadd double %232, %240
  br label %fused_add_real_53_5_clone_1.exit88

242:                                              ; preds = %207
  %243 = zext nneg i32 %24 to i64
  %244 = getelementptr inbounds [16 x i8], ptr addrspace(1) %12, i64 %243
  %245 = load <2 x double>, ptr addrspace(1) %244, align 16, !invariant.load !9, !alias.scope !190, !noalias !195
  %.unpack.i.i12.i.i.i58295 = extractelement <2 x double> %245, i32 0
  %.unpack2.i.i14.i.i.i60296 = extractelement <2 x double> %245, i32 1
  %246 = fmul double %35, %.unpack.i.i12.i.i.i58295
  %247 = fmul double %36, %.unpack2.i.i14.i.i.i60296
  %248 = fsub double %246, %247
  %249 = fmul double %35, %.unpack2.i.i14.i.i.i60296
  %250 = fmul double %36, %.unpack.i.i12.i.i.i58295
  %251 = fadd double %250, %249
  br label %fused_add_real_53_5_clone_1.exit88

fused_add_real_53_5_clone_1.exit88:               ; preds = %209, %242
  %.unpack2.i.i.i.i.i112 = phi double [ %.unpack2.i.i14.i.i.i60296, %242 ], [ %.unpack2.i.i.i.i.i81288, %209 ]
  %.unpack.i.i.i.i.i110 = phi double [ %.unpack.i.i12.i.i.i58295, %242 ], [ %.unpack.i.i.i.i.i79287, %209 ]
  %252 = phi double [ %248, %242 ], [ %237, %209 ]
  %253 = phi double [ %251, %242 ], [ %241, %209 ]
  %254 = icmp ult i32 %24, 62
  %255 = zext nneg i32 %23 to i64
  %256 = getelementptr inbounds [16 x i8], ptr addrspace(1) %12, i64 %255
  %257 = load <2 x double>, ptr addrspace(1) %256, align 16, !invariant.load !9, !alias.scope !200, !noalias !203
  %.unpack.i.i.i69285 = extractelement <2 x double> %257, i32 0
  %.unpack3.i.i.i68286 = extractelement <2 x double> %257, i32 1
  %258 = fmul double %252, %.unpack.i.i.i69285
  %259 = fmul double %253, %.unpack3.i.i.i68286
  %260 = fadd double %259, %258
  tail call void @llvm.experimental.noalias.scope.decl(metadata !204)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !207)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !209)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !211)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !214)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !216)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !218)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !221)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !223)
  br i1 %254, label %261, label %293

261:                                              ; preds = %fused_add_real_53_5_clone_1.exit88
  %262 = add nuw nsw i32 %24, 2
  %263 = zext nneg i32 %24 to i64
  %264 = getelementptr inbounds [16 x i8], ptr addrspace(1) %12, i64 %263
  %265 = getelementptr inbounds i8, ptr addrspace(1) %264, i64 1056
  %266 = load <2 x double>, ptr addrspace(1) %265, align 16, !invariant.load !9, !alias.scope !225, !noalias !226
  %.unpack.i1.i.i101279 = extractelement <2 x double> %266, i32 0
  %.unpack2.i.i.i103280 = extractelement <2 x double> %266, i32 1
  tail call void @llvm.experimental.noalias.scope.decl(metadata !227)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !230)
  %267 = zext nneg i32 %25 to i64
  %268 = add i64 %267, %263
  %269 = getelementptr inbounds [16 x i8], ptr addrspace(1) %10, i64 %268
  %270 = getelementptr inbounds i8, ptr addrspace(1) %269, i64 32
  %271 = load <2 x double>, ptr addrspace(1) %270, align 16, !invariant.load !9, !alias.scope !232, !noalias !233
  %.unpack.i.i.i.i104281 = extractelement <2 x double> %271, i32 0
  %.unpack2.i.i.i.i106282 = extractelement <2 x double> %271, i32 1
  %272 = shl nuw nsw i32 %262, 6
  %273 = or disjoint i32 %272, %23
  %274 = zext nneg i32 %273 to i64
  %275 = getelementptr inbounds [16 x i8], ptr addrspace(1) %11, i64 %274
  %276 = load <2 x double>, ptr addrspace(1) %275, align 16, !invariant.load !9, !alias.scope !234, !noalias !235
  %.unpack3.i.i.i.i107283 = extractelement <2 x double> %276, i32 0
  %.unpack5.i.i.i.i109284 = extractelement <2 x double> %276, i32 1
  %277 = fadd double %.unpack.i.i.i.i104281, %.unpack3.i.i.i.i107283
  %278 = fadd double %.unpack2.i.i.i.i106282, %.unpack5.i.i.i.i109284
  %279 = fmul double %.unpack.i1.i.i101279, %277
  %280 = fmul double %.unpack2.i.i.i103280, %278
  %281 = fmul double %.unpack2.i.i.i103280, %277
  %282 = fmul double %.unpack.i1.i.i101279, %278
  %283 = fsub double %279, %280
  %284 = fadd double %281, %282
  %285 = fmul double %35, %.unpack.i.i.i.i.i110
  %286 = fmul double %36, %.unpack2.i.i.i.i.i112
  %287 = fsub double %285, %286
  %288 = fadd double %283, %287
  %289 = fmul double %35, %.unpack2.i.i.i.i.i112
  %290 = fmul double %36, %.unpack.i.i.i.i.i110
  %291 = fadd double %290, %289
  %292 = fadd double %284, %291
  br label %fused_add_imag_43_5_clone_1.exit119

293:                                              ; preds = %fused_add_real_53_5_clone_1.exit88
  %294 = fmul double %35, %.unpack.i.i.i.i.i110
  %295 = fmul double %36, %.unpack2.i.i.i.i.i112
  %296 = fsub double %294, %295
  %297 = fmul double %35, %.unpack2.i.i.i.i.i112
  %298 = fmul double %36, %.unpack.i.i.i.i.i110
  %299 = fadd double %298, %297
  br label %fused_add_imag_43_5_clone_1.exit119

fused_add_imag_43_5_clone_1.exit119:              ; preds = %261, %293
  %300 = phi double [ %296, %293 ], [ %288, %261 ]
  %301 = phi double [ %299, %293 ], [ %292, %261 ]
  %302 = fmul double %.unpack.i.i.i69285, %301
  %303 = fmul double %.unpack3.i.i.i68286, %300
  %304 = fsub double %302, %303
  br label %305

305:                                              ; preds = %fused_add_imag_43_5_clone_1.exit119, %fused_add_imag_43_5_clone_1.exit
  %306 = phi double [ %260, %fused_add_imag_43_5_clone_1.exit119 ], [ %160, %fused_add_imag_43_5_clone_1.exit ]
  %307 = phi double [ %304, %fused_add_imag_43_5_clone_1.exit119 ], [ %206, %fused_add_imag_43_5_clone_1.exit ]
  %.not4 = icmp eq i32 %23, 63
  br i1 %.not4, label %345, label %308

308:                                              ; preds = %305
  %309 = add nuw nsw i32 %23, 1
  %.not5 = icmp eq i32 %24, 63
  br i1 %.not5, label %332, label %310

310:                                              ; preds = %308
  %311 = zext nneg i32 %24 to i64
  %312 = getelementptr inbounds [16 x i8], ptr addrspace(1) %14, i64 %311
  %313 = getelementptr inbounds i8, ptr addrspace(1) %312, i64 16
  %314 = load <2 x double>, ptr addrspace(1) %313, align 16, !invariant.load !9
  %.unpack273 = extractelement <2 x double> %314, i32 0
  %.unpack7274 = extractelement <2 x double> %314, i32 1
  tail call void @llvm.experimental.noalias.scope.decl(metadata !236)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !239)
  %315 = shl nuw nsw i32 %309, 6
  %316 = zext nneg i32 %315 to i64
  %317 = add i64 %316, %311
  %318 = getelementptr inbounds [16 x i8], ptr addrspace(1) %10, i64 %317
  %319 = getelementptr inbounds i8, ptr addrspace(1) %318, i64 16
  %320 = load <2 x double>, ptr addrspace(1) %319, align 16, !invariant.load !9, !alias.scope !236, !noalias !239
  %.unpack.i120275 = extractelement <2 x double> %320, i32 0
  %.unpack2.i122276 = extractelement <2 x double> %320, i32 1
  %321 = getelementptr inbounds i8, ptr addrspace(1) %33, i64 1024
  %322 = getelementptr inbounds i8, ptr addrspace(1) %321, i64 16
  %323 = load <2 x double>, ptr addrspace(1) %322, align 16, !invariant.load !9, !alias.scope !239, !noalias !236
  %.unpack3.i123277 = extractelement <2 x double> %323, i32 0
  %.unpack5.i125278 = extractelement <2 x double> %323, i32 1
  %324 = fadd double %.unpack.i120275, %.unpack3.i123277
  %325 = fadd double %.unpack2.i122276, %.unpack5.i125278
  %326 = fmul double %.unpack273, %324
  %327 = fmul double %.unpack7274, %325
  %328 = fmul double %.unpack7274, %324
  %329 = fmul double %.unpack273, %325
  %330 = fsub double %326, %327
  %331 = fadd double %328, %329
  br label %332

332:                                              ; preds = %310, %308
  %333 = phi double [ %330, %310 ], [ 0.000000e+00, %308 ]
  %334 = phi double [ %331, %310 ], [ 0.000000e+00, %308 ]
  %335 = zext nneg i32 %23 to i64
  %336 = getelementptr inbounds [16 x i8], ptr addrspace(1) %14, i64 %335
  %337 = getelementptr inbounds i8, ptr addrspace(1) %336, i64 16
  %338 = load <2 x double>, ptr addrspace(1) %337, align 16, !invariant.load !9
  %.unpack8271 = extractelement <2 x double> %338, i32 0
  %.unpack10272 = extractelement <2 x double> %338, i32 1
  %339 = fmul double %333, %.unpack8271
  %340 = fmul double %334, %.unpack8271
  %341 = fmul double %334, %.unpack10272
  %342 = fadd double %339, %341
  %343 = fmul double %333, %.unpack10272
  %344 = fsub double %340, %343
  br label %345

345:                                              ; preds = %332, %305
  %346 = phi double [ %342, %332 ], [ 0.000000e+00, %305 ]
  %347 = phi double [ %344, %332 ], [ 0.000000e+00, %305 ]
  %348 = icmp samesign ult i32 %24, 62
  br i1 %348, label %349, label %fused_add_add_417_3_clone_1.exit193

349:                                              ; preds = %345
  %350 = add nuw nsw i32 %24, 2
  %351 = zext nneg i32 %24 to i64
  %352 = getelementptr inbounds [16 x i8], ptr addrspace(1) %15, i64 %351
  %353 = getelementptr inbounds i8, ptr addrspace(1) %352, i64 2080
  %354 = load <2 x double>, ptr addrspace(1) %353, align 16, !invariant.load !9, !alias.scope !241, !noalias !244
  %.unpack.i.i126255 = extractelement <2 x double> %354, i32 0
  %.unpack2.i.i128256 = extractelement <2 x double> %354, i32 1
  %355 = fadd double %.unpack.i.i126255, 0.000000e+00
  %356 = fadd double %.unpack2.i.i128256, 0.000000e+00
  tail call void @llvm.experimental.noalias.scope.decl(metadata !246)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !249)
  %357 = zext nneg i32 %25 to i64
  %358 = add i64 %357, %351
  %359 = getelementptr inbounds [16 x i8], ptr addrspace(1) %10, i64 %358
  %360 = getelementptr inbounds i8, ptr addrspace(1) %359, i64 32
  %361 = load <2 x double>, ptr addrspace(1) %360, align 16, !invariant.load !9, !alias.scope !246, !noalias !249
  %.unpack.i129257 = extractelement <2 x double> %361, i32 0
  %.unpack2.i131258 = extractelement <2 x double> %361, i32 1
  %362 = shl nuw nsw i32 %350, 6
  %363 = or disjoint i32 %362, %23
  %364 = zext nneg i32 %363 to i64
  %365 = getelementptr inbounds [16 x i8], ptr addrspace(1) %11, i64 %364
  %366 = load <2 x double>, ptr addrspace(1) %365, align 16, !invariant.load !9, !alias.scope !249, !noalias !246
  %.unpack3.i132259 = extractelement <2 x double> %366, i32 0
  %.unpack5.i134260 = extractelement <2 x double> %366, i32 1
  %367 = fadd double %.unpack.i129257, %.unpack3.i132259
  %368 = fadd double %.unpack2.i131258, %.unpack5.i134260
  %369 = fmul double %355, %367
  %370 = fmul double %356, %368
  %371 = fmul double %356, %367
  %372 = fmul double %355, %368
  %373 = fsub double %369, %370
  %374 = fadd double %371, %372
  tail call void @llvm.experimental.noalias.scope.decl(metadata !251)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !254)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !256)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !258)
  %375 = icmp samesign ugt i32 %24, 1
  br i1 %375, label %376, label %fused_add_add_417_3_clone_1.exit

376:                                              ; preds = %349
  %377 = add nsw i32 %24, -2
  tail call void @llvm.experimental.noalias.scope.decl(metadata !260)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !263)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !265)
  %378 = zext nneg i32 %377 to i64
  %379 = getelementptr inbounds [16 x i8], ptr addrspace(1) %15, i64 %378
  %380 = load <2 x double>, ptr addrspace(1) %379, align 16, !invariant.load !9, !alias.scope !267, !noalias !270
  %.unpack.i1.i.i.i.i.i249 = extractelement <2 x double> %380, i32 0
  %.unpack2.i2.i.i.i.i.i250 = extractelement <2 x double> %380, i32 1
  tail call void @llvm.experimental.noalias.scope.decl(metadata !272)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !275)
  %381 = add nuw nsw i32 %25, %377
  %382 = zext nneg i32 %381 to i64
  %383 = getelementptr inbounds [16 x i8], ptr addrspace(1) %10, i64 %382
  %384 = load <2 x double>, ptr addrspace(1) %383, align 16, !invariant.load !9, !alias.scope !277, !noalias !278
  %.unpack.i.i.i141251 = extractelement <2 x double> %384, i32 0
  %.unpack2.i.i.i143252 = extractelement <2 x double> %384, i32 1
  %385 = shl nuw nsw i32 %377, 6
  %386 = or disjoint i32 %385, %23
  %387 = zext nneg i32 %386 to i64
  %388 = getelementptr inbounds [16 x i8], ptr addrspace(1) %11, i64 %387
  %389 = load <2 x double>, ptr addrspace(1) %388, align 16, !invariant.load !9, !alias.scope !279, !noalias !280
  %.unpack3.i.i.i144253 = extractelement <2 x double> %389, i32 0
  %.unpack5.i.i.i254 = extractelement <2 x double> %389, i32 1
  %390 = fadd double %.unpack.i.i.i141251, %.unpack3.i.i.i144253
  %391 = fadd double %.unpack2.i.i.i143252, %.unpack5.i.i.i254
  %392 = fmul double %.unpack.i1.i.i.i.i.i249, %390
  %393 = fmul double %.unpack2.i2.i.i.i.i.i250, %391
  %394 = fsub double %392, %393
  br label %fused_add_add_417_3_clone_1.exit

fused_add_add_417_3_clone_1.exit:                 ; preds = %349, %376
  %395 = phi double [ %394, %376 ], [ 0.000000e+00, %349 ]
  %396 = icmp ugt i32 %24, 1
  tail call void @llvm.experimental.noalias.scope.decl(metadata !281)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !284)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !286)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !289)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !291)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !294)
  %397 = getelementptr inbounds [16 x i8], ptr addrspace(1) %16, i64 %351
  %398 = load <2 x double>, ptr addrspace(1) %397, align 16, !invariant.load !9, !alias.scope !296, !noalias !301
  %.unpack.i.i.i.i.i.i135245 = extractelement <2 x double> %398, i32 0
  %.unpack2.i.i.i.i.i.i137246 = extractelement <2 x double> %398, i32 1
  %399 = fmul double %.unpack.i.i.i.i.i.i135245, 0.000000e+00
  %400 = fsub double %.unpack2.i.i.i.i.i.i137246, %399
  %401 = getelementptr inbounds i8, ptr addrspace(1) %352, i64 1024
  %402 = load <2 x double>, ptr addrspace(1) %401, align 16, !invariant.load !9, !alias.scope !305, !noalias !306
  %.unpack.i.i.i.i.i138247 = extractelement <2 x double> %402, i32 0
  %.unpack2.i.i.i.i.i140248 = extractelement <2 x double> %402, i32 1
  %403 = fadd double %400, %.unpack.i.i.i.i.i138247
  %404 = fmul double %.unpack2.i.i.i.i.i.i137246, -0.000000e+00
  %405 = fsub double %404, %.unpack.i.i.i.i.i.i135245
  %406 = fadd double %.unpack2.i.i.i.i.i140248, %405
  %407 = fmul double %403, %35
  %408 = fmul double %406, %36
  %409 = fsub double %407, %408
  %410 = fadd double %395, %409
  %411 = fadd double %373, %410
  tail call void @llvm.experimental.noalias.scope.decl(metadata !307)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !310)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !312)
  br i1 %396, label %412, label %fused_add_add_419_3_clone_1.exit

412:                                              ; preds = %fused_add_add_417_3_clone_1.exit
  %413 = add nsw i32 %24, -2
  tail call void @llvm.experimental.noalias.scope.decl(metadata !314)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !317)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !319)
  %414 = zext nneg i32 %413 to i64
  %415 = getelementptr inbounds [16 x i8], ptr addrspace(1) %15, i64 %414
  %416 = load <2 x double>, ptr addrspace(1) %415, align 16, !invariant.load !9, !alias.scope !321, !noalias !324
  %.unpack.i1.i.i.i.i.i160239 = extractelement <2 x double> %416, i32 0
  %.unpack2.i2.i.i.i.i.i162240 = extractelement <2 x double> %416, i32 1
  tail call void @llvm.experimental.noalias.scope.decl(metadata !327)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !330)
  %417 = add nuw nsw i32 %25, %413
  %418 = zext nneg i32 %417 to i64
  %419 = getelementptr inbounds [16 x i8], ptr addrspace(1) %10, i64 %418
  %420 = load <2 x double>, ptr addrspace(1) %419, align 16, !invariant.load !9, !alias.scope !332, !noalias !333
  %.unpack.i.i.i163241 = extractelement <2 x double> %420, i32 0
  %.unpack2.i.i.i165242 = extractelement <2 x double> %420, i32 1
  %421 = shl nuw nsw i32 %413, 6
  %422 = or disjoint i32 %421, %23
  %423 = zext nneg i32 %422 to i64
  %424 = getelementptr inbounds [16 x i8], ptr addrspace(1) %11, i64 %423
  %425 = load <2 x double>, ptr addrspace(1) %424, align 16, !invariant.load !9, !alias.scope !334, !noalias !335
  %.unpack3.i.i.i166243 = extractelement <2 x double> %425, i32 0
  %.unpack5.i.i.i168244 = extractelement <2 x double> %425, i32 1
  %426 = fadd double %.unpack.i.i.i163241, %.unpack3.i.i.i166243
  %427 = fadd double %.unpack2.i.i.i165242, %.unpack5.i.i.i168244
  %428 = fmul double %.unpack2.i2.i.i.i.i.i162240, %426
  %429 = fmul double %.unpack.i1.i.i.i.i.i160239, %427
  %430 = fadd double %428, %429
  br label %fused_add_add_419_3_clone_1.exit

fused_add_add_419_3_clone_1.exit:                 ; preds = %fused_add_add_417_3_clone_1.exit, %412
  %431 = phi double [ %430, %412 ], [ 0.000000e+00, %fused_add_add_417_3_clone_1.exit ]
  %432 = fmul double %406, %35
  %433 = fmul double %403, %36
  %434 = fadd double %432, %433
  %435 = fadd double %431, %434
  %436 = fadd double %374, %435
  br label %478

fused_add_add_417_3_clone_1.exit193:              ; preds = %345
  tail call void @llvm.experimental.noalias.scope.decl(metadata !336)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !339)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !341)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !343)
  %437 = add nsw i32 %24, -2
  tail call void @llvm.experimental.noalias.scope.decl(metadata !345)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !348)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !350)
  %438 = zext nneg i32 %437 to i64
  %439 = getelementptr inbounds [16 x i8], ptr addrspace(1) %15, i64 %438
  %440 = load <2 x double>, ptr addrspace(1) %439, align 16, !invariant.load !9, !alias.scope !352, !noalias !355
  %.unpack.i1.i.i.i.i.i184263 = extractelement <2 x double> %440, i32 0
  %.unpack2.i2.i.i.i.i.i186264 = extractelement <2 x double> %440, i32 1
  tail call void @llvm.experimental.noalias.scope.decl(metadata !357)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !360)
  %441 = add nuw nsw i32 %25, %437
  %442 = zext nneg i32 %441 to i64
  %443 = getelementptr inbounds [16 x i8], ptr addrspace(1) %10, i64 %442
  %444 = load <2 x double>, ptr addrspace(1) %443, align 16, !invariant.load !9, !alias.scope !362, !noalias !363
  %.unpack.i.i.i187265 = extractelement <2 x double> %444, i32 0
  %.unpack2.i.i.i189266 = extractelement <2 x double> %444, i32 1
  %445 = shl nuw nsw i32 %437, 6
  %446 = or disjoint i32 %445, %23
  %447 = zext nneg i32 %446 to i64
  %448 = getelementptr inbounds [16 x i8], ptr addrspace(1) %11, i64 %447
  %449 = load <2 x double>, ptr addrspace(1) %448, align 16, !invariant.load !9, !alias.scope !364, !noalias !365
  %.unpack3.i.i.i190267 = extractelement <2 x double> %449, i32 0
  %.unpack5.i.i.i192268 = extractelement <2 x double> %449, i32 1
  %450 = fadd double %.unpack.i.i.i187265, %.unpack3.i.i.i190267
  %451 = fadd double %.unpack2.i.i.i189266, %.unpack5.i.i.i192268
  %452 = fmul double %.unpack.i1.i.i.i.i.i184263, %450
  %453 = fmul double %.unpack2.i2.i.i.i.i.i186264, %451
  %454 = fsub double %452, %453
  tail call void @llvm.experimental.noalias.scope.decl(metadata !366)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !369)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !371)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !374)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !376)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !379)
  %455 = zext nneg i32 %24 to i64
  %456 = getelementptr inbounds [16 x i8], ptr addrspace(1) %16, i64 %455
  %457 = load <2 x double>, ptr addrspace(1) %456, align 16, !invariant.load !9, !alias.scope !381, !noalias !386
  %.unpack.i.i.i.i.i.i172269 = extractelement <2 x double> %457, i32 0
  %.unpack2.i.i.i.i.i.i174270 = extractelement <2 x double> %457, i32 1
  %458 = fmul double %.unpack.i.i.i.i.i.i172269, 0.000000e+00
  %459 = fsub double %.unpack2.i.i.i.i.i.i174270, %458
  %460 = getelementptr inbounds [16 x i8], ptr addrspace(1) %15, i64 %455
  %461 = getelementptr inbounds i8, ptr addrspace(1) %460, i64 1024
  %462 = load <2 x double>, ptr addrspace(1) %461, align 16, !invariant.load !9, !alias.scope !390, !noalias !391
  %.unpack.i.i.i.i.i175261 = extractelement <2 x double> %462, i32 0
  %.unpack2.i.i.i.i.i177262 = extractelement <2 x double> %462, i32 1
  %463 = fadd double %459, %.unpack.i.i.i.i.i175261
  %464 = fmul double %.unpack2.i.i.i.i.i.i174270, -0.000000e+00
  %465 = fsub double %464, %.unpack.i.i.i.i.i.i172269
  %466 = fadd double %.unpack2.i.i.i.i.i177262, %465
  %467 = fmul double %463, %35
  %468 = fmul double %466, %36
  %469 = fsub double %467, %468
  %470 = fadd double %454, %469
  %471 = fmul double %.unpack2.i2.i.i.i.i.i186264, %450
  %472 = fmul double %.unpack.i1.i.i.i.i.i184263, %451
  %473 = fadd double %471, %472
  %474 = fmul double %466, %35
  %475 = fmul double %463, %36
  %476 = fadd double %474, %475
  %477 = fadd double %473, %476
  br label %478

478:                                              ; preds = %fused_add_add_417_3_clone_1.exit193, %fused_add_add_419_3_clone_1.exit
  %479 = phi double [ %470, %fused_add_add_417_3_clone_1.exit193 ], [ %411, %fused_add_add_419_3_clone_1.exit ]
  %480 = phi double [ %477, %fused_add_add_417_3_clone_1.exit193 ], [ %436, %fused_add_add_419_3_clone_1.exit ]
  %481 = fadd double %306, %346
  %482 = fadd double %307, %347
  %483 = fmul double %481, 5.000000e-01
  %484 = fmul double %482, 0.000000e+00
  %485 = fmul double %482, 5.000000e-01
  %486 = fmul double %481, 0.000000e+00
  %487 = fsub double %483, %484
  %488 = fadd double %486, %485
  %489 = fadd double %487, %479
  %490 = fadd double %488, %480
  %491 = shl nuw nsw i32 %19, 7
  %492 = or disjoint i32 %491, %20
  %493 = zext nneg i32 %492 to i64
  %494 = getelementptr inbounds [16 x i8], ptr addrspace(1) %17, i64 %493
  %495 = insertelement <2 x double> poison, double %35, i32 0
  %496 = insertelement <2 x double> %495, double %36, i32 1
  store <2 x double> %496, ptr addrspace(1) %494, align 16
  %497 = getelementptr inbounds [16 x i8], ptr addrspace(1) %18, i64 %493
  %498 = insertelement <2 x double> poison, double %489, i32 0
  %499 = insertelement <2 x double> %498, double %490, i32 1
  store <2 x double> %499, ptr addrspace(1) %497, align 16
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
!5 = distinct !{!5, !6, !"fused_add_add_416_1: argument 0"}
!6 = distinct !{!6, !"fused_add_add_416_1"}
!7 = !{!8}
!8 = distinct !{!8, !6, !"fused_add_add_416_1: argument 1"}
!9 = !{}
!10 = !{!11}
!11 = distinct !{!11, !12, !"fused_add_complex_36_16_clone_1: argument 0"}
!12 = distinct !{!12, !"fused_add_complex_36_16_clone_1"}
!13 = !{!14}
!14 = distinct !{!14, !12, !"fused_add_complex_36_16_clone_1: argument 1"}
!15 = !{!16}
!16 = distinct !{!16, !12, !"fused_add_complex_36_16_clone_1: argument 2"}
!17 = !{!11, !14}
!18 = !{!19}
!19 = distinct !{!19, !20, !"fused_add_add_416_1: argument 0"}
!20 = distinct !{!20, !"fused_add_add_416_1"}
!21 = !{!22}
!22 = distinct !{!22, !20, !"fused_add_add_416_1: argument 1"}
!23 = !{!19, !11}
!24 = !{!22, !14, !16}
!25 = !{!22, !14}
!26 = !{!19, !11, !16}
!27 = !{!28}
!28 = distinct !{!28, !29, !"fused_add_real_51_5_clone_1: argument 0"}
!29 = distinct !{!29, !"fused_add_real_51_5_clone_1"}
!30 = !{!31}
!31 = distinct !{!31, !29, !"fused_add_real_51_5_clone_1: argument 1"}
!32 = !{!33}
!33 = distinct !{!33, !29, !"fused_add_real_51_5_clone_1: argument 2"}
!34 = !{!35}
!35 = distinct !{!35, !36, !"fused_add_multiply_82_9_clone_1: argument 0"}
!36 = distinct !{!36, !"fused_add_multiply_82_9_clone_1"}
!37 = !{!38}
!38 = distinct !{!38, !36, !"fused_add_multiply_82_9_clone_1: argument 1"}
!39 = !{!40}
!40 = distinct !{!40, !36, !"fused_add_multiply_82_9_clone_1: argument 2"}
!41 = !{!40, !33, !16}
!42 = !{!35, !38, !28, !31, !11, !14}
!43 = !{!44}
!44 = distinct !{!44, !45, !"fused_add_add_416_1: argument 0"}
!45 = distinct !{!45, !"fused_add_add_416_1"}
!46 = !{!47}
!47 = distinct !{!47, !45, !"fused_add_add_416_1: argument 1"}
!48 = !{!44, !35, !28, !11}
!49 = !{!47, !38, !40, !31, !33, !14, !16}
!50 = !{!47, !38, !31, !14}
!51 = !{!44, !35, !40, !28, !33, !11, !16}
!52 = !{!53}
!53 = distinct !{!53, !54, !"fused_add_real_51_5_clone_1: argument 0"}
!54 = distinct !{!54, !"fused_add_real_51_5_clone_1"}
!55 = !{!56}
!56 = distinct !{!56, !54, !"fused_add_real_51_5_clone_1: argument 1"}
!57 = !{!58}
!58 = distinct !{!58, !54, !"fused_add_real_51_5_clone_1: argument 2"}
!59 = !{!60}
!60 = distinct !{!60, !61, !"fused_add_multiply_82_9_clone_1: argument 0"}
!61 = distinct !{!61, !"fused_add_multiply_82_9_clone_1"}
!62 = !{!63}
!63 = distinct !{!63, !61, !"fused_add_multiply_82_9_clone_1: argument 1"}
!64 = !{!65}
!65 = distinct !{!65, !61, !"fused_add_multiply_82_9_clone_1: argument 2"}
!66 = !{!65, !58, !16}
!67 = !{!60, !63, !53, !56, !11, !14}
!68 = !{!69}
!69 = distinct !{!69, !70, !"fused_add_add_416_1: argument 0"}
!70 = distinct !{!70, !"fused_add_add_416_1"}
!71 = !{!72}
!72 = distinct !{!72, !70, !"fused_add_add_416_1: argument 1"}
!73 = !{!69, !60, !53, !11}
!74 = !{!72, !63, !65, !56, !58, !14, !16}
!75 = !{!72, !63, !56, !14}
!76 = !{!69, !60, !65, !53, !58, !11, !16}
!77 = !{!78}
!78 = distinct !{!78, !79, !"fused_add_complex_32_16_clone_1: argument 0"}
!79 = distinct !{!79, !"fused_add_complex_32_16_clone_1"}
!80 = !{!81}
!81 = distinct !{!81, !82, !"fused_add_real_53_5_clone_1: argument 0"}
!82 = distinct !{!82, !"fused_add_real_53_5_clone_1"}
!83 = !{!84}
!84 = distinct !{!84, !82, !"fused_add_real_53_5_clone_1: argument 1"}
!85 = !{!86}
!86 = distinct !{!86, !82, !"fused_add_real_53_5_clone_1: argument 2"}
!87 = !{!88}
!88 = distinct !{!88, !89, !"fused_add_multiply_84_9_clone_1: argument 0"}
!89 = distinct !{!89, !"fused_add_multiply_84_9_clone_1"}
!90 = !{!91}
!91 = distinct !{!91, !89, !"fused_add_multiply_84_9_clone_1: argument 1"}
!92 = !{!93}
!93 = distinct !{!93, !89, !"fused_add_multiply_84_9_clone_1: argument 2"}
!94 = !{!95}
!95 = distinct !{!95, !96, !"fused_add_complex_36_16_clone_1: argument 0"}
!96 = distinct !{!96, !"fused_add_complex_36_16_clone_1"}
!97 = !{!98}
!98 = distinct !{!98, !96, !"fused_add_complex_36_16_clone_1: argument 1"}
!99 = !{!100}
!100 = distinct !{!100, !96, !"fused_add_complex_36_16_clone_1: argument 2"}
!101 = !{!100, !93, !86}
!102 = !{!95, !98, !88, !91, !81, !84}
!103 = !{!104}
!104 = distinct !{!104, !105, !"fused_add_add_416_1: argument 0"}
!105 = distinct !{!105, !"fused_add_add_416_1"}
!106 = !{!107}
!107 = distinct !{!107, !105, !"fused_add_add_416_1: argument 1"}
!108 = !{!104, !95, !88, !81}
!109 = !{!107, !98, !100, !91, !93, !84, !86}
!110 = !{!107, !98, !91, !84}
!111 = !{!104, !95, !100, !88, !93, !81, !86}
!112 = !{!113, !93, !86}
!113 = distinct !{!113, !114, !"fused_add_complex_32_16_clone_1: argument 0"}
!114 = distinct !{!114, !"fused_add_complex_32_16_clone_1"}
!115 = !{!88, !91, !81, !84}
!116 = !{!117}
!117 = distinct !{!117, !118, !"fused_add_imag_43_5_clone_1: argument 0"}
!118 = distinct !{!118, !"fused_add_imag_43_5_clone_1"}
!119 = !{!120}
!120 = distinct !{!120, !118, !"fused_add_imag_43_5_clone_1: argument 1"}
!121 = !{!122}
!122 = distinct !{!122, !118, !"fused_add_imag_43_5_clone_1: argument 2"}
!123 = !{!124}
!124 = distinct !{!124, !125, !"fused_add_multiply_84_9_clone_1: argument 0"}
!125 = distinct !{!125, !"fused_add_multiply_84_9_clone_1"}
!126 = !{!127}
!127 = distinct !{!127, !125, !"fused_add_multiply_84_9_clone_1: argument 1"}
!128 = !{!129}
!129 = distinct !{!129, !125, !"fused_add_multiply_84_9_clone_1: argument 2"}
!130 = !{!131}
!131 = distinct !{!131, !132, !"fused_add_complex_36_16_clone_1: argument 0"}
!132 = distinct !{!132, !"fused_add_complex_36_16_clone_1"}
!133 = !{!134}
!134 = distinct !{!134, !132, !"fused_add_complex_36_16_clone_1: argument 1"}
!135 = !{!136}
!136 = distinct !{!136, !132, !"fused_add_complex_36_16_clone_1: argument 2"}
!137 = !{!136, !129, !122}
!138 = !{!131, !134, !124, !127, !117, !120}
!139 = !{!140}
!140 = distinct !{!140, !141, !"fused_add_add_416_1: argument 0"}
!141 = distinct !{!141, !"fused_add_add_416_1"}
!142 = !{!143}
!143 = distinct !{!143, !141, !"fused_add_add_416_1: argument 1"}
!144 = !{!140, !131, !124, !117}
!145 = !{!143, !134, !136, !127, !129, !120, !122}
!146 = !{!143, !134, !127, !120}
!147 = !{!140, !131, !136, !124, !129, !117, !122}
!148 = !{!149}
!149 = distinct !{!149, !150, !"fused_add_real_53_5_clone_1: argument 0"}
!150 = distinct !{!150, !"fused_add_real_53_5_clone_1"}
!151 = !{!152}
!152 = distinct !{!152, !150, !"fused_add_real_53_5_clone_1: argument 1"}
!153 = !{!154}
!154 = distinct !{!154, !150, !"fused_add_real_53_5_clone_1: argument 2"}
!155 = !{!156}
!156 = distinct !{!156, !157, !"fused_add_multiply_84_9_clone_1: argument 0"}
!157 = distinct !{!157, !"fused_add_multiply_84_9_clone_1"}
!158 = !{!159}
!159 = distinct !{!159, !157, !"fused_add_multiply_84_9_clone_1: argument 1"}
!160 = !{!161}
!161 = distinct !{!161, !157, !"fused_add_multiply_84_9_clone_1: argument 2"}
!162 = !{!163}
!163 = distinct !{!163, !164, !"fused_add_complex_36_16_clone_1: argument 0"}
!164 = distinct !{!164, !"fused_add_complex_36_16_clone_1"}
!165 = !{!166}
!166 = distinct !{!166, !164, !"fused_add_complex_36_16_clone_1: argument 1"}
!167 = !{!168}
!168 = distinct !{!168, !164, !"fused_add_complex_36_16_clone_1: argument 2"}
!169 = !{!168, !161, !154}
!170 = !{!163, !166, !156, !159, !149, !152}
!171 = !{!172}
!172 = distinct !{!172, !173, !"fused_add_add_416_1: argument 0"}
!173 = distinct !{!173, !"fused_add_add_416_1"}
!174 = !{!175}
!175 = distinct !{!175, !173, !"fused_add_add_416_1: argument 1"}
!176 = !{!172, !163, !156, !149}
!177 = !{!175, !166, !168, !159, !161, !152, !154}
!178 = !{!175, !166, !159, !152}
!179 = !{!172, !163, !168, !156, !161, !149, !154}
!180 = !{!181, !183, !168, !161, !154}
!181 = distinct !{!181, !182, !"fused_add_multiply_82_9_clone_1: argument 2"}
!182 = distinct !{!182, !"fused_add_multiply_82_9_clone_1"}
!183 = distinct !{!183, !184, !"fused_add_real_51_5_clone_1: argument 2"}
!184 = distinct !{!184, !"fused_add_real_51_5_clone_1"}
!185 = !{!186, !187, !188, !189, !163, !166, !156, !159, !149, !152}
!186 = distinct !{!186, !182, !"fused_add_multiply_82_9_clone_1: argument 0"}
!187 = distinct !{!187, !182, !"fused_add_multiply_82_9_clone_1: argument 1"}
!188 = distinct !{!188, !184, !"fused_add_real_51_5_clone_1: argument 0"}
!189 = distinct !{!189, !184, !"fused_add_real_51_5_clone_1: argument 1"}
!190 = !{!191, !193, !168, !161, !154}
!191 = distinct !{!191, !192, !"fused_add_multiply_82_9_clone_1: argument 2"}
!192 = distinct !{!192, !"fused_add_multiply_82_9_clone_1"}
!193 = distinct !{!193, !194, !"fused_add_real_51_5_clone_1: argument 2"}
!194 = distinct !{!194, !"fused_add_real_51_5_clone_1"}
!195 = !{!196, !197, !198, !199, !163, !166, !156, !159, !149, !152}
!196 = distinct !{!196, !192, !"fused_add_multiply_82_9_clone_1: argument 0"}
!197 = distinct !{!197, !192, !"fused_add_multiply_82_9_clone_1: argument 1"}
!198 = distinct !{!198, !194, !"fused_add_real_51_5_clone_1: argument 0"}
!199 = distinct !{!199, !194, !"fused_add_real_51_5_clone_1: argument 1"}
!200 = !{!201, !161, !154}
!201 = distinct !{!201, !202, !"fused_add_complex_32_16_clone_1: argument 0"}
!202 = distinct !{!202, !"fused_add_complex_32_16_clone_1"}
!203 = !{!156, !159, !149, !152}
!204 = !{!205}
!205 = distinct !{!205, !206, !"fused_add_imag_43_5_clone_1: argument 0"}
!206 = distinct !{!206, !"fused_add_imag_43_5_clone_1"}
!207 = !{!208}
!208 = distinct !{!208, !206, !"fused_add_imag_43_5_clone_1: argument 1"}
!209 = !{!210}
!210 = distinct !{!210, !206, !"fused_add_imag_43_5_clone_1: argument 2"}
!211 = !{!212}
!212 = distinct !{!212, !213, !"fused_add_multiply_84_9_clone_1: argument 0"}
!213 = distinct !{!213, !"fused_add_multiply_84_9_clone_1"}
!214 = !{!215}
!215 = distinct !{!215, !213, !"fused_add_multiply_84_9_clone_1: argument 1"}
!216 = !{!217}
!217 = distinct !{!217, !213, !"fused_add_multiply_84_9_clone_1: argument 2"}
!218 = !{!219}
!219 = distinct !{!219, !220, !"fused_add_complex_36_16_clone_1: argument 0"}
!220 = distinct !{!220, !"fused_add_complex_36_16_clone_1"}
!221 = !{!222}
!222 = distinct !{!222, !220, !"fused_add_complex_36_16_clone_1: argument 1"}
!223 = !{!224}
!224 = distinct !{!224, !220, !"fused_add_complex_36_16_clone_1: argument 2"}
!225 = !{!224, !217, !210}
!226 = !{!219, !222, !212, !215, !205, !208}
!227 = !{!228}
!228 = distinct !{!228, !229, !"fused_add_add_416_1: argument 0"}
!229 = distinct !{!229, !"fused_add_add_416_1"}
!230 = !{!231}
!231 = distinct !{!231, !229, !"fused_add_add_416_1: argument 1"}
!232 = !{!228, !219, !212, !205}
!233 = !{!231, !222, !224, !215, !217, !208, !210}
!234 = !{!231, !222, !215, !208}
!235 = !{!228, !219, !224, !212, !217, !205, !210}
!236 = !{!237}
!237 = distinct !{!237, !238, !"fused_add_add_416_1: argument 0"}
!238 = distinct !{!238, !"fused_add_add_416_1"}
!239 = !{!240}
!240 = distinct !{!240, !238, !"fused_add_add_416_1: argument 1"}
!241 = !{!242}
!242 = distinct !{!242, !243, !"fused_add_scatter_add_646_7_clone_1: argument 0"}
!243 = distinct !{!243, !"fused_add_scatter_add_646_7_clone_1"}
!244 = !{!245}
!245 = distinct !{!245, !243, !"fused_add_scatter_add_646_7_clone_1: argument 1"}
!246 = !{!247}
!247 = distinct !{!247, !248, !"fused_add_add_416_1: argument 0"}
!248 = distinct !{!248, !"fused_add_add_416_1"}
!249 = !{!250}
!250 = distinct !{!250, !248, !"fused_add_add_416_1: argument 1"}
!251 = !{!252}
!252 = distinct !{!252, !253, !"fused_add_add_417_3_clone_1: argument 0"}
!253 = distinct !{!253, !"fused_add_add_417_3_clone_1"}
!254 = !{!255}
!255 = distinct !{!255, !253, !"fused_add_add_417_3_clone_1: argument 1"}
!256 = !{!257}
!257 = distinct !{!257, !253, !"fused_add_add_417_3_clone_1: argument 2"}
!258 = !{!259}
!259 = distinct !{!259, !253, !"fused_add_add_417_3_clone_1: argument 3"}
!260 = !{!261}
!261 = distinct !{!261, !262, !"fused_add_multiply_79_13_clone_1: argument 0"}
!262 = distinct !{!262, !"fused_add_multiply_79_13_clone_1"}
!263 = !{!264}
!264 = distinct !{!264, !262, !"fused_add_multiply_79_13_clone_1: argument 1"}
!265 = !{!266}
!266 = distinct !{!266, !262, !"fused_add_multiply_79_13_clone_1: argument 2"}
!267 = !{!268, !266, !257}
!268 = distinct !{!268, !269, !"fused_add_scatter_add_646_7_clone_1: argument 0"}
!269 = distinct !{!269, !"fused_add_scatter_add_646_7_clone_1"}
!270 = !{!271, !261, !264, !252, !255, !259}
!271 = distinct !{!271, !269, !"fused_add_scatter_add_646_7_clone_1: argument 1"}
!272 = !{!273}
!273 = distinct !{!273, !274, !"fused_add_add_416_1: argument 0"}
!274 = distinct !{!274, !"fused_add_add_416_1"}
!275 = !{!276}
!276 = distinct !{!276, !274, !"fused_add_add_416_1: argument 1"}
!277 = !{!273, !261, !252}
!278 = !{!276, !264, !266, !255, !257, !259}
!279 = !{!276, !264, !255}
!280 = !{!273, !261, !266, !252, !257, !259}
!281 = !{!282}
!282 = distinct !{!282, !283, !"fused_add_multiply_80_13_clone_1: argument 2"}
!283 = distinct !{!283, !"fused_add_multiply_80_13_clone_1"}
!284 = !{!285}
!285 = distinct !{!285, !283, !"fused_add_multiply_80_13_clone_1: argument 3"}
!286 = !{!287}
!287 = distinct !{!287, !288, !"fused_add_scatter_add_646_7_clone_1: argument 0"}
!288 = distinct !{!288, !"fused_add_scatter_add_646_7_clone_1"}
!289 = !{!290}
!290 = distinct !{!290, !288, !"fused_add_scatter_add_646_7_clone_1: argument 1"}
!291 = !{!292}
!292 = distinct !{!292, !293, !"fused_add_dynamic_update_slice_225_8_clone_1: argument 0"}
!293 = distinct !{!293, !"fused_add_dynamic_update_slice_225_8_clone_1"}
!294 = !{!295}
!295 = distinct !{!295, !293, !"fused_add_dynamic_update_slice_225_8_clone_1: argument 1"}
!296 = !{!297, !299, !295, !290, !285, !259}
!297 = distinct !{!297, !298, !"fused_add_mul_384_3_clone_1: argument 0"}
!298 = distinct !{!298, !"fused_add_mul_384_3_clone_1"}
!299 = distinct !{!299, !300, !"fused_add_dynamic_update_slice_224_9_clone_1: argument 1"}
!300 = distinct !{!300, !"fused_add_dynamic_update_slice_224_9_clone_1"}
!301 = !{!302, !292, !287, !303, !304, !282, !252, !255, !257}
!302 = distinct !{!302, !300, !"fused_add_dynamic_update_slice_224_9_clone_1: argument 0"}
!303 = distinct !{!303, !283, !"fused_add_multiply_80_13_clone_1: argument 0"}
!304 = distinct !{!304, !283, !"fused_add_multiply_80_13_clone_1: argument 1"}
!305 = !{!287, !282, !257}
!306 = !{!290, !303, !304, !285, !252, !255, !259}
!307 = !{!308}
!308 = distinct !{!308, !309, !"fused_add_add_419_3_clone_1: argument 0"}
!309 = distinct !{!309, !"fused_add_add_419_3_clone_1"}
!310 = !{!311}
!311 = distinct !{!311, !309, !"fused_add_add_419_3_clone_1: argument 1"}
!312 = !{!313}
!313 = distinct !{!313, !309, !"fused_add_add_419_3_clone_1: argument 2"}
!314 = !{!315}
!315 = distinct !{!315, !316, !"fused_add_multiply_79_13_clone_1: argument 0"}
!316 = distinct !{!316, !"fused_add_multiply_79_13_clone_1"}
!317 = !{!318}
!318 = distinct !{!318, !316, !"fused_add_multiply_79_13_clone_1: argument 1"}
!319 = !{!320}
!320 = distinct !{!320, !316, !"fused_add_multiply_79_13_clone_1: argument 2"}
!321 = !{!322, !320, !313}
!322 = distinct !{!322, !323, !"fused_add_scatter_add_646_7_clone_1: argument 0"}
!323 = distinct !{!323, !"fused_add_scatter_add_646_7_clone_1"}
!324 = !{!325, !315, !318, !308, !311, !326}
!325 = distinct !{!325, !323, !"fused_add_scatter_add_646_7_clone_1: argument 1"}
!326 = distinct !{!326, !309, !"fused_add_add_419_3_clone_1: argument 3"}
!327 = !{!328}
!328 = distinct !{!328, !329, !"fused_add_add_416_1: argument 0"}
!329 = distinct !{!329, !"fused_add_add_416_1"}
!330 = !{!331}
!331 = distinct !{!331, !329, !"fused_add_add_416_1: argument 1"}
!332 = !{!328, !315, !308}
!333 = !{!331, !318, !320, !311, !313, !326}
!334 = !{!331, !318, !311}
!335 = !{!328, !315, !320, !308, !313, !326}
!336 = !{!337}
!337 = distinct !{!337, !338, !"fused_add_add_417_3_clone_1: argument 0"}
!338 = distinct !{!338, !"fused_add_add_417_3_clone_1"}
!339 = !{!340}
!340 = distinct !{!340, !338, !"fused_add_add_417_3_clone_1: argument 1"}
!341 = !{!342}
!342 = distinct !{!342, !338, !"fused_add_add_417_3_clone_1: argument 2"}
!343 = !{!344}
!344 = distinct !{!344, !338, !"fused_add_add_417_3_clone_1: argument 3"}
!345 = !{!346}
!346 = distinct !{!346, !347, !"fused_add_multiply_79_13_clone_1: argument 0"}
!347 = distinct !{!347, !"fused_add_multiply_79_13_clone_1"}
!348 = !{!349}
!349 = distinct !{!349, !347, !"fused_add_multiply_79_13_clone_1: argument 1"}
!350 = !{!351}
!351 = distinct !{!351, !347, !"fused_add_multiply_79_13_clone_1: argument 2"}
!352 = !{!353, !351, !342}
!353 = distinct !{!353, !354, !"fused_add_scatter_add_646_7_clone_1: argument 0"}
!354 = distinct !{!354, !"fused_add_scatter_add_646_7_clone_1"}
!355 = !{!356, !346, !349, !337, !340, !344}
!356 = distinct !{!356, !354, !"fused_add_scatter_add_646_7_clone_1: argument 1"}
!357 = !{!358}
!358 = distinct !{!358, !359, !"fused_add_add_416_1: argument 0"}
!359 = distinct !{!359, !"fused_add_add_416_1"}
!360 = !{!361}
!361 = distinct !{!361, !359, !"fused_add_add_416_1: argument 1"}
!362 = !{!358, !346, !337}
!363 = !{!361, !349, !351, !340, !342, !344}
!364 = !{!361, !349, !340}
!365 = !{!358, !346, !351, !337, !342, !344}
!366 = !{!367}
!367 = distinct !{!367, !368, !"fused_add_multiply_80_13_clone_1: argument 2"}
!368 = distinct !{!368, !"fused_add_multiply_80_13_clone_1"}
!369 = !{!370}
!370 = distinct !{!370, !368, !"fused_add_multiply_80_13_clone_1: argument 3"}
!371 = !{!372}
!372 = distinct !{!372, !373, !"fused_add_scatter_add_646_7_clone_1: argument 0"}
!373 = distinct !{!373, !"fused_add_scatter_add_646_7_clone_1"}
!374 = !{!375}
!375 = distinct !{!375, !373, !"fused_add_scatter_add_646_7_clone_1: argument 1"}
!376 = !{!377}
!377 = distinct !{!377, !378, !"fused_add_dynamic_update_slice_225_8_clone_1: argument 0"}
!378 = distinct !{!378, !"fused_add_dynamic_update_slice_225_8_clone_1"}
!379 = !{!380}
!380 = distinct !{!380, !378, !"fused_add_dynamic_update_slice_225_8_clone_1: argument 1"}
!381 = !{!382, !384, !380, !375, !370, !344}
!382 = distinct !{!382, !383, !"fused_add_mul_384_3_clone_1: argument 0"}
!383 = distinct !{!383, !"fused_add_mul_384_3_clone_1"}
!384 = distinct !{!384, !385, !"fused_add_dynamic_update_slice_224_9_clone_1: argument 1"}
!385 = distinct !{!385, !"fused_add_dynamic_update_slice_224_9_clone_1"}
!386 = !{!387, !377, !372, !388, !389, !367, !337, !340, !342}
!387 = distinct !{!387, !385, !"fused_add_dynamic_update_slice_224_9_clone_1: argument 0"}
!388 = distinct !{!388, !368, !"fused_add_multiply_80_13_clone_1: argument 0"}
!389 = distinct !{!389, !368, !"fused_add_multiply_80_13_clone_1: argument 1"}
!390 = !{!372, !367, !342}
!391 = !{!375, !388, !389, !370, !337, !340, !344}
