// code by jph
package ch.ethz.idsc.subare.core.util;

import java.awt.Dimension;
import java.awt.Point;
import java.util.List;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.util.color.Colorscheme;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Array;
import ch.ethz.idsc.tensor.alg.Dimensions;
import ch.ethz.idsc.tensor.alg.Join;
import ch.ethz.idsc.tensor.opt.Interpolation;

public enum StateActionRasters {
  ;
  // ---
  private static final Interpolation COLORSCHEME = Colorscheme.classic();
  private static final Tensor BASE = Tensors.vector(255);

  /** @param stateActionRaster
   * @param qsa scaled to contain values in the interval [0, 1]
   * @return */
  public static Tensor render(StateActionRaster stateActionRaster, DiscreteQsa qsa) {
    DiscreteModel discreteModel = stateActionRaster.discreteModel();
    Dimension dimension = stateActionRaster.dimension();
    final Tensor tensor = Array.zeros(dimension.width, dimension.height, 4);
    for (Tensor state : discreteModel.states())
      for (Tensor action : discreteModel.actions(state)) {
        Point point = stateActionRaster.point(state, action);
        if (point != null) {
          Scalar sca = qsa.value(state, action);
          tensor.set(COLORSCHEME.get(BASE.multiply(sca)), point.x, point.y);
        }
      }
    return tensor;
  }

  public static Tensor render(StateActionRaster stateActionRaster, PolicyInterface policyInterface) {
    return render(stateActionRaster, Policies.toQsa(stateActionRaster.discreteModel(), policyInterface));
  }

  public static Tensor qsaPolicy(StateActionRaster stateActionRaster, DiscreteQsa qsa) {
    Tensor image1 = render(stateActionRaster, TensorValuesUtils.rescaled(qsa));
    PolicyInterface policyInterface = GreedyPolicy.bestEquiprobable(stateActionRaster.discreteModel(), qsa);
    Tensor image2 = render(stateActionRaster, policyInterface);
    List<Integer> list = Dimensions.of(image1);
    list.set(0, 3);
    return Join.of(0, image1, Array.zeros(list), image2);
  }

  public static Tensor qsaPolicyRef(StateActionRaster stateActionRaster, DiscreteQsa qsa, DiscreteQsa ref) {
    Tensor image1 = render(stateActionRaster, TensorValuesUtils.rescaled(qsa));
    PolicyInterface policyInterface = GreedyPolicy.bestEquiprobable(stateActionRaster.discreteModel(), qsa);
    Tensor image2 = render(stateActionRaster, policyInterface);
    // TODO magic const
    Tensor image3 = render(stateActionRaster, TensorValuesUtils.logisticDifference(qsa, ref, RealScalar.of(15)));
    List<Integer> list = Dimensions.of(image1);
    list.set(0, 3);
    return Join.of(0, image1, Array.zeros(list), image2, Array.zeros(list), image3);
  }

  public static Tensor qsaRef(StateActionRaster stateActionRaster, DiscreteQsa qsa, DiscreteQsa ref) {
    Tensor image1 = render(stateActionRaster, TensorValuesUtils.rescaled(qsa));
    Tensor image2 = render(stateActionRaster, TensorValuesUtils.logisticDifference(qsa, ref));
    List<Integer> list = Dimensions.of(image1);
    list.set(0, 3);
    return Join.of(0, image1, Array.zeros(list), image2);
  }
}
