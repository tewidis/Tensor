/* 
 * This file is part of Tensor.
 * 
 * Tensor is free software: you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License as published by the Free Software 
 * Foundation, either version 3 of the License, or any later version.
 * 
 * Tensor is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR * A PARTICULAR PURPOSE. See the GNU General Public License for more 
 * details.
 * 
 * You should have received a copy of the GNU General Public License along with 
 * Tensor. If not, see <https://www.gnu.org/licenses/>. 
 */

#pragma once

namespace gt
{
    enum OPERATION {
        PLUS,
        MINUS,
        TIMES,
        DIVIDE,
        POWER,
        MAX,
        MIN,
        MOD,
        REM,
        ATAN2,
        ATAN2D,
        HYPOT
    };

    enum CONVOLUTION {
        FULL,
        SAME,
        VALID
    };
};
