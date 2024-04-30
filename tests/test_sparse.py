"""
Test Functions for the SparseDIA class
"""

from dynamiqs.sparse import *
import jax
import jax.numpy as jnp



class TestSparseDIA:

    def test_matmul(self, matrixA, matrixB):
        sparseA = to_sparse(matrixA)
        sparseB = to_sparse(matrixB)

        out_dia_dia = sparseA @ sparseB
        out_dia_dense = sparseA @ matrixB
        out_dense_dense = matrixA @ matrixB

        if jnp.allclose(out_dia_dense, out_dense_dense) and jnp.allclose(out_dia_dia.to_dense(), out_dense_dense):
            print("Dia Matmul works!")
        else:
            print("Results not matching")

    def test_add(self, matrixA, matrixB):

        sparseA = to_sparse(matrixA)
        sparseB = to_sparse(matrixB)

        out_dia_dia = sparseA + sparseB
        out_dia_dense = sparseA + matrixB
        out_dense_dense = matrixA + matrixB

        if jnp.allclose(out_dia_dense, out_dense_dense) and jnp.allclose(out_dia_dia.to_dense(), out_dense_dense):
            print("Dia Addition works!")
        else:
            print("Results not matching!")

    def test_transform(self, matrix):

        if jnp.allclose(matrix, to_sparse(matrix).to_dense()):
            print("Transform works!")
        else:
            print("Transform is not working!")
