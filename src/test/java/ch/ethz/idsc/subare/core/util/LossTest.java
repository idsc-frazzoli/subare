// code by jph
package ch.ethz.idsc.subare.core.util;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.tensor.RationalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import junit.framework.TestCase;

public class LossTest extends TestCase {
  static DiscreteModel create14() {
    return new DiscreteModel() {
      @Override
      public Scalar gamma() {
        return null;
      }

      @Override
      public Tensor states() {
        return Tensors.vector(0);
      }

      @Override
      public Tensor actions(Tensor state) {
        return Tensors.vector(0, 1, 2, 3);
      }
    };
  }

  public void testAccumulation0() {
    DiscreteModel discreteModel = create14();
    DiscreteQsa ref = DiscreteQsa.build(discreteModel).create(Tensors.vector(0, 0, 0, 0).stream());
    DiscreteQsa qsa = DiscreteQsa.build(discreteModel).create(Tensors.vector(0, 0, 0, 0).stream());
    Scalar scalar = Loss.accumulation(discreteModel, ref, qsa);
    assertEquals(scalar, RealScalar.ZERO);
  }

  public void testAccumulation1() {
    DiscreteModel discreteModel = create14();
    DiscreteQsa ref = DiscreteQsa.build(discreteModel).create(Tensors.vector(0, 0, 1, 0).stream());
    DiscreteQsa qsa = DiscreteQsa.build(discreteModel).create(Tensors.vector(1, 0, 2, -2).stream());
    Scalar scalar = Loss.accumulation(discreteModel, ref, qsa);
    assertEquals(scalar, RealScalar.ZERO);
  }

  public void testAccumulation2() {
    DiscreteModel discreteModel = create14();
    DiscreteQsa ref = DiscreteQsa.build(discreteModel).create(Tensors.vector(0, 0, 1, 0).stream());
    DiscreteQsa qsa = DiscreteQsa.build(discreteModel).create(Tensors.vector(2, 0, 2, -2).stream());
    Scalar scalar = Loss.accumulation(discreteModel, ref, qsa);
    assertEquals(scalar, RationalScalar.of(1, 2));
  }

  public void testAccumulation3() {
    DiscreteModel discreteModel = create14();
    DiscreteQsa ref = DiscreteQsa.build(discreteModel).create(Tensors.vector(0, 0, 1, 0).stream());
    DiscreteQsa qsa = DiscreteQsa.build(discreteModel).create(Tensors.vector(2, 2, 1.5, 2).stream());
    Scalar scalar = Loss.accumulation(discreteModel, ref, qsa);
    assertEquals(scalar, RealScalar.ONE);
  }
}
