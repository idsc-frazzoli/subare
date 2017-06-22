// code by jph
package ch.ethz.idsc.subare.core.util.gfx;

import java.awt.Dimension;
import java.awt.Point;
import java.util.List;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteValueFunctions;
import ch.ethz.idsc.subare.core.util.GreedyPolicy;
import ch.ethz.idsc.subare.core.util.Loss;
import ch.ethz.idsc.subare.core.util.Policies;
import ch.ethz.idsc.subare.util.Colorscheme;
import ch.ethz.idsc.subare.util.ImageResize;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Array;
import ch.ethz.idsc.tensor.alg.Dimensions;
import ch.ethz.idsc.tensor.alg.Join;
import ch.ethz.idsc.tensor.opt.Interpolation;
import ch.ethz.idsc.tensor.sca.Clip;

// TODO all non-terminal function should be package visibility
public enum StateActionRasters {
  ;
  private static final Interpolation COLORSCHEME = Colorscheme.classic();
  private static final Tensor BASE = Tensors.vector(255);

  /** @param stateActionRaster
   * @param qsa scaled to contain values in the interval [0, 1]
   * @return */
  public static Tensor render(StateActionRaster stateActionRaster, DiscreteQsa qsa) {
    DiscreteModel discreteModel = stateActionRaster.discreteModel();
    Dimension dimension = stateActionRaster.dimensionStateActionRaster();
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

  private static Tensor render(StateActionRaster stateActionRaster, Policy policy) {
    return render(stateActionRaster, Policies.toQsa(stateActionRaster.discreteModel(), policy));
  }

  /***************************************************/
  public static Tensor qsa(StateActionRaster stateActionRaster, DiscreteQsa qsa) {
    return ImageResize.of(render(stateActionRaster, qsa), stateActionRaster.magify());
  }

  public static Tensor qsaPolicy(StateActionRaster stateActionRaster, DiscreteQsa qsa) {
    Tensor image1 = render(stateActionRaster, DiscreteValueFunctions.rescaled(qsa));
    Policy policy = GreedyPolicy.bestEquiprobable(stateActionRaster.discreteModel(), qsa);
    Tensor image2 = render(stateActionRaster, policy);
    List<Integer> list = Dimensions.of(image1);
    list.set(0, 3);
    return ImageResize.of( //
        Join.of(0, image1, Array.zeros(list), image2), stateActionRaster.magify());
  }

  public static Tensor qsaPolicyRef(StateActionRaster stateActionRaster, DiscreteQsa qsa, DiscreteQsa ref) {
    Tensor image1 = render(stateActionRaster, DiscreteValueFunctions.rescaled(qsa));
    Policy policy = GreedyPolicy.bestEquiprobable(stateActionRaster.discreteModel(), qsa);
    Tensor image2 = render(stateActionRaster, policy);
    Scalar qdelta = stateActionRaster.scaleQdelta();
    Tensor image3 = render(stateActionRaster, DiscreteValueFunctions.logisticDifference(qsa, ref, qdelta));
    List<Integer> list = Dimensions.of(image1);
    list.set(0, 3);
    return ImageResize.of( //
        Join.of(0, image1, Array.zeros(list), image2, Array.zeros(list), image3), stateActionRaster.magify());
  }

  public static Tensor qsaLossRef(StateActionRaster stateActionRaster, DiscreteQsa qsa, DiscreteQsa ref) {
    Tensor image1 = render(stateActionRaster, DiscreteValueFunctions.rescaled(qsa));
    DiscreteQsa loss = Loss.asQsa(stateActionRaster.discreteModel(), ref, qsa);
    loss = loss.create(loss.values().flatten(0) //
        .map(tensor -> tensor.multiply(stateActionRaster.scaleLoss())) //
        .map(Clip.UNIT::of));
    Tensor image2 = render(stateActionRaster, loss);
    Tensor image3 = render(stateActionRaster, DiscreteValueFunctions.logisticDifference(qsa, ref));
    List<Integer> list = Dimensions.of(image1);
    int dim = stateActionRaster.joinAlongDimension();
    list.set(dim, 1);
    return ImageResize.of( //
        Join.of(dim, image1, Array.zeros(list), image2, Array.zeros(list), image3), stateActionRaster.magify());
  }

  // not recommended, use qsaLossRef instead
  static Tensor qsaRef(StateActionRaster stateActionRaster, DiscreteQsa qsa, DiscreteQsa ref) {
    Tensor image1 = render(stateActionRaster, DiscreteValueFunctions.rescaled(qsa));
    Scalar qdelta = stateActionRaster.scaleQdelta();
    Tensor image2 = render(stateActionRaster, DiscreteValueFunctions.logisticDifference(qsa, ref, qdelta));
    List<Integer> list = Dimensions.of(image1);
    int dim = stateActionRaster.joinAlongDimension();
    list.set(dim, 3);
    return ImageResize.of( //
        Join.of(0, image1, Array.zeros(list), image2), stateActionRaster.magify());
  }
}
